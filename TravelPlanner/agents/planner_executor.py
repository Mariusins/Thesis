"""Planner-Executor two-agent system for RQ1.

Pipeline per query:
  1. Planner LLM call (no tools) -> JSON plan (sub-goals, budget allocation).
  2. Executor: ReactAgent (upstream) with the plan injected into its prompt;
     same 7 tools, same 30-step cap, same NotebookWrite, same final Planner[Query]
     hand-off to produce the natural-language plan.

If the planner fails to emit valid JSON twice, fall through to a pure ReAct run
so the query still produces an answer (degraded mode flagged in the result).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator

from llm_client import LLMClient, TokenLedger
from prompts_planner import (
    EXECUTOR_PLAN_PREAMBLE,
    PLANNER_FEW_SHOT_EXAMPLE,
    PLANNER_JSON_SCHEMA,
    PLANNER_SYSTEM_PROMPT,
)
from tool_agents import ReactAgent

_VALIDATOR = Draft7Validator(PLANNER_JSON_SCHEMA)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from LLM output.

    Models sometimes wrap JSON in ```json fences or add trailing prose.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class PlannerAgent:
    """Stateless LLM call that emits a sub-goal JSON plan."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        ledger: Optional[TokenLedger] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.ledger = ledger if ledger is not None else TokenLedger()
        self.seed = seed
        self.llm = LLMClient(
            model=model,
            temperature=0.0,
            max_tokens=2048,
            ledger=self.ledger,
        )

    def plan(self, query_record: Dict[str, Any]) -> Dict[str, Any]:
        """Returns dict with keys: plan (or None), raw, valid (bool), errors (list)."""
        sys_prompt = PLANNER_SYSTEM_PROMPT.format(
            org=query_record.get("org"),
            dest=query_record.get("dest"),
            days=query_record.get("days"),
            date=query_record.get("date"),
            people_number=query_record.get("people_number"),
            local_constraint=query_record.get("local_constraint"),
            budget=query_record.get("budget"),
            query=query_record.get("query"),
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Produce the JSON plan now. Reference example:\n" + PLANNER_FEW_SHOT_EXAMPLE},
        ]
        last_raw = ""
        last_errors: List[str] = []
        for attempt in range(2):
            raw = self.llm.chat(
                messages,
                seed=self.seed,
                role_tag="planner_agent",
                response_format={"type": "json_object"},
            )
            last_raw = raw
            obj = _extract_json(raw)
            if obj is None:
                last_errors = ["json-parse-failed"]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "That was not valid JSON. Output only the JSON object, no fences, no prose."})
                continue
            errors = [e.message for e in _VALIDATOR.iter_errors(obj)]
            if not errors:
                return {"plan": obj, "raw": raw, "valid": True, "errors": []}
            last_errors = errors
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Schema errors: " + "; ".join(errors) + "\nFix and re-emit the JSON object only."})
        return {"plan": None, "raw": last_raw, "valid": False, "errors": last_errors}


class PlannerExecutorAgent:
    """Composes PlannerAgent + ReactAgent (executor) over a shared TokenLedger."""

    DEFAULT_TOOLS = [
        "notebook",
        "flights",
        "attractions",
        "accommodations",
        "restaurants",
        "googleDistanceMatrix",
        "planner",
        "cities",
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        seed: Optional[int] = None,
        max_steps: int = 30,
    ) -> None:
        self.ledger = TokenLedger()
        self.seed = seed
        self.model = model
        self.planner = PlannerAgent(model=model, ledger=self.ledger, seed=seed)
        self.executor = ReactAgent(
            None,
            tools=list(self.DEFAULT_TOOLS),
            max_steps=max_steps,
            react_llm_name=model,
            planner_llm_name=model,
            ledger=self.ledger,
            seed=seed,
        )

    def run(self, query_record: Dict[str, Any]):
        plan_result = self.planner.plan(query_record)
        plan_obj = plan_result["plan"]

        if plan_obj is None:
            # Degraded: run pure ReAct without plan injection.
            self.executor.agent_prompt.template = self._original_template_or(self.executor.agent_prompt.template)
            answer, scratchpad, action_log = self.executor.run(query_record["query"])
            return {
                "answer": answer,
                "scratchpad": scratchpad,
                "action_log": action_log,
                "plan_json": None,
                "plan_raw": plan_result["raw"],
                "plan_valid": False,
                "plan_errors": plan_result["errors"],
                "ledger": self.ledger.snapshot(),
                "degraded": True,
            }

        # Inject plan into executor's agent_prompt template.
        # The downstream prompt is consumed via str.format(query=..., scratchpad=...),
        # so any literal braces in the JSON plan must be escaped to {{ and }}.
        plan_str = json.dumps(plan_obj, indent=2).replace("{", "{{").replace("}", "}}")
        original = self.executor.agent_prompt.template
        if not getattr(self.executor.agent_prompt, "_original_template", None):
            self.executor.agent_prompt._original_template = original
        injected = EXECUTOR_PLAN_PREAMBLE.format(plan_json=plan_str) + self.executor.agent_prompt._original_template
        self.executor.agent_prompt.template = injected

        try:
            answer, scratchpad, action_log = self.executor.run(query_record["query"])
        finally:
            self.executor.agent_prompt.template = self.executor.agent_prompt._original_template

        return {
            "answer": answer,
            "scratchpad": scratchpad,
            "action_log": action_log,
            "plan_json": plan_obj,
            "plan_raw": plan_result["raw"],
            "plan_valid": True,
            "plan_errors": [],
            "ledger": self.ledger.snapshot(),
            "degraded": False,
        }

    @staticmethod
    def _original_template_or(template: str) -> str:
        return template
