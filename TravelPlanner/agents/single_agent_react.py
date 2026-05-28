"""Single-agent ReAct baseline for RQ1 (and reused by RQ2 SC / BC wrappers).

Thin shim around upstream ReactAgent. Identical tool set, identical prompt,
identical 30-step cap. The point is to share inference path (LLMClient) with
the planner-executor so seeds and token accounting are comparable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from llm_client import TokenLedger
from tool_agents import ReactAgent


class SingleAgentReact:
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
        ledger: Optional[TokenLedger] = None,
        token_budget: Optional[int] = None,
    ) -> None:
        self.ledger = ledger if ledger is not None else TokenLedger()
        self.seed = seed
        self.model = model
        self.agent = ReactAgent(
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
        answer, scratchpad, action_log = self.agent.run(query_record["query"])
        return {
            "answer": answer,
            "scratchpad": scratchpad,
            "action_log": action_log,
            "ledger": self.ledger.snapshot(),
        }
