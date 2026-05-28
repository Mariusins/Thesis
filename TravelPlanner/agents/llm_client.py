"""Modern OpenAI SDK wrapper used by every LLM call site in this fork.

Single entry point so single-agent and planner-executor share identical
inference paths and token accounting. Returns (content, usage) where usage
is a dict with prompt_tokens, completion_tokens, total_tokens.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError, BadRequestError


@dataclass
class TokenLedger:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_calls: int = 0
    by_role: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, usage: Dict[str, int], role: str = "default") -> None:
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.n_calls += 1
        slot = self.by_role.setdefault(role, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "n_calls": 0})
        slot["prompt_tokens"] += usage.get("prompt_tokens", 0)
        slot["completion_tokens"] += usage.get("completion_tokens", 0)
        slot["total_tokens"] += usage.get("total_tokens", 0)
        slot["n_calls"] += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "n_calls": self.n_calls,
            "by_role": dict(self.by_role),
        }


class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
        ledger: Optional[TokenLedger] = None,
        max_retries: int = 10,
        base_url: Optional[str] = None,
    ) -> None:
        # base_url precedence: explicit arg > OPENAI_BASE_URL env > OpenAI default.
        # When pointing at a local vLLM server, OPENAI_API_KEY is not required;
        # the SDK still demands a non-empty key, so we pass a sentinel.
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if url:
            key = key or "EMPTY"
        elif not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ledger = ledger if ledger is not None else TokenLedger()
        self.max_retries = max_retries

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        role_tag: str = "default",
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if stop:
            kwargs["stop"] = stop
        if response_format is not None:
            kwargs["response_format"] = response_format

        backoff = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                if usage is not None:
                    self.ledger.add(
                        {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        },
                        role=role_tag,
                    )
                return content
            except RateLimitError:
                time.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
            except (APIConnectionError, APIError) as e:
                if attempt + 1 == self.max_retries:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
            except AuthenticationError:
                raise
            except BadRequestError:
                raise
        raise RuntimeError("LLMClient.chat exhausted retries")

    # Compatibility shim for legacy code that wraps content in a HumanMessage list.
    def __call__(self, prompt: str, *, role_tag: str = "default", stop: Optional[List[str]] = None) -> "_LegacyResponse":
        text = self.chat(
            [{"role": "user", "content": prompt}],
            stop=stop,
            role_tag=role_tag,
        )
        return _LegacyResponse(text)


class _LegacyResponse:
    """Mimics langchain's `.content` attribute so minimal-diff migrations work."""
    def __init__(self, content: str) -> None:
        self.content = content
