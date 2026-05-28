"""Budget-capped single-agent ReAct baseline for RQ2.

Plain SingleAgentReact with raised max_steps and a token-budget early-halt so
it can use up to the matched PE budget (~52k tokens) without runaway. The
budget enforcement lives in ReactAgent.is_halted via the token_budget plumbing.
"""
from __future__ import annotations

from typing import Optional

from single_agent_react import SingleAgentReact


class BudgetCappedReactAgent(SingleAgentReact):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        seed: Optional[int] = None,
        max_steps: int = 50,
        token_budget: Optional[int] = 52000,
    ) -> None:
        super().__init__(
            model=model,
            seed=seed,
            max_steps=max_steps,
            token_budget=token_budget,
        )
