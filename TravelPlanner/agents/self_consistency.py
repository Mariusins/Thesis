"""Self-consistency wrapper around SingleAgentReact for RQ2.

Runs N independent SingleAgentReact samples with shifted seeds, sharing one
TokenLedger so the combined budget is comparable to PE. Picks the first sample
whose answer is non-empty and not the upstream "Max Token Length Exceeded."
sentinel. All samples are logged for post-hoc analysis.

Token-budget halt: before starting sample i+1, check the shared ledger total
against `token_budget` and break if exceeded. Per-call halting inside a single
sample is handled by ReactAgent via the same `token_budget` plumbing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_client import TokenLedger
from single_agent_react import SingleAgentReact


_INVALID_ANSWER_SENTINELS = {"", "Max Token Length Exceeded."}


def _is_valid(answer: Any) -> bool:
    if answer is None:
        return False
    if isinstance(answer, str) and answer.strip() in _INVALID_ANSWER_SENTINELS:
        return False
    return True


class SelfConsistencyAgent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        seed: Optional[int] = None,
        max_steps: int = 30,
        n_samples: int = 2,
        token_budget: Optional[int] = None,
    ) -> None:
        self.ledger = TokenLedger()
        self.model = model
        self.seed = seed if seed is not None else 0
        self.max_steps = max_steps
        self.n_samples = n_samples
        self.token_budget = token_budget

    def run(self, query_record: Dict[str, Any]):
        samples: List[Dict[str, Any]] = []
        winning: Optional[Dict[str, Any]] = None
        winner_idx: Optional[int] = None
        last_out: Optional[Dict[str, Any]] = None
        last_i: Optional[int] = None

        for i in range(self.n_samples):
            if i > 0 and self.token_budget is not None and self.ledger.total_tokens >= self.token_budget:
                break

            sample_seed = self.seed + i * 1000
            tokens_before = self.ledger.total_tokens
            agent = SingleAgentReact(
                model=self.model,
                seed=sample_seed,
                max_steps=self.max_steps,
                ledger=self.ledger,
                token_budget=self.token_budget,
            )
            out = agent.run(query_record)
            tokens_after = self.ledger.total_tokens
            last_out = out
            last_i = i

            samples.append({
                "sample_idx": i,
                "sample_seed": sample_seed,
                "answer": out["answer"],
                "tokens_used": tokens_after - tokens_before,
                "tokens_total_after": tokens_after,
            })

            if winning is None and _is_valid(out["answer"]):
                winning = out
                winner_idx = i

        if winning is None:
            winning = last_out
            winner_idx = last_i

        return {
            "answer": winning["answer"] if winning else None,
            "scratchpad": winning["scratchpad"] if winning else "",
            "action_log": winning["action_log"] if winning else [],
            "ledger": self.ledger.snapshot(),
            "samples": samples,
            "n_samples_run": len(samples),
            "winner_idx": winner_idx,
        }
