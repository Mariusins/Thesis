"""Plan-then-execute single-agent baseline for RQ2.

Differs from RQ1's PlannerExecutor:
  - same model identity for plan + execute (no Planner role separation)
  - free-text plan, not JSON-schema-validated
  - one planning call followed by a ReAct executor whose system prompt is
    prefixed with the plan
Token ledger is shared so the matched-compute comparison is exact.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from llm_client import LLMClient, TokenLedger
from prompts_pte import PTE_EXECUTOR_PREAMBLE, PTE_PLAN_PROMPT
from tool_agents import ReactAgent


class PlanThenExecuteAgent:
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
        token_budget: Optional[int] = None,
    ) -> None:
        self.ledger = TokenLedger()
        self.model = model
        self.seed = seed
        self.token_budget = token_budget
        self.plan_llm = LLMClient(
            model=model,
            temperature=0.0,
            max_tokens=1024,
            ledger=self.ledger,
        )
        self.executor = ReactAgent(
            None,
            tools=list(self.DEFAULT_TOOLS),
            max_steps=max_steps,
            react_llm_name=model,
            planner_llm_name=model,
            ledger=self.ledger,
            seed=seed,
            token_budget=token_budget,
        )

    def run(self, query_record: Dict[str, Any]):
        plan_prompt = PTE_PLAN_PROMPT.format(
            org=query_record.get("org"),
            dest=query_record.get("dest"),
            days=query_record.get("days"),
            date=query_record.get("date"),
            people_number=query_record.get("people_number"),
            local_constraint=query_record.get("local_constraint"),
            budget=query_record.get("budget"),
            query=query_record.get("query"),
        )
        plan_text = self.plan_llm.chat(
            [{"role": "user", "content": plan_prompt}],
            seed=self.seed,
            role_tag="pte_plan",
        )

        # Inject plan into executor prompt template. Escape literal braces in the
        # plan text so str.format(query=, scratchpad=) downstream doesn't choke.
        safe_plan = plan_text.replace("{", "{{").replace("}", "}}")
        original = self.executor.agent_prompt.template
        if not getattr(self.executor.agent_prompt, "_original_template", None):
            self.executor.agent_prompt._original_template = original
        injected = PTE_EXECUTOR_PREAMBLE.format(plan_text=safe_plan) + self.executor.agent_prompt._original_template
        self.executor.agent_prompt.template = injected

        try:
            answer, scratchpad, action_log = self.executor.run(query_record["query"])
        finally:
            self.executor.agent_prompt.template = self.executor.agent_prompt._original_template

        return {
            "answer": answer,
            "scratchpad": scratchpad,
            "action_log": action_log,
            "plan_text": plan_text,
            "ledger": self.ledger.snapshot(),
        }
