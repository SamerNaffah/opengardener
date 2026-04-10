"""
Unified LLM client — provider-agnostic interface.

Set LLM_PROVIDER in the environment:
  - "ollama"    (default, fully local, no API key needed)
  - "anthropic" (requires ANTHROPIC_API_KEY)
  - "openai"    (requires OPENAI_API_KEY)

Set LLM_MODEL to choose the model:
  - ollama:    "llama3.2" (default), "mistral", "llama3.1", etc.
  - anthropic: "claude-haiku-4-5-20251001", "claude-sonnet-4-6", etc.
  - openai:    "gpt-4o-mini", "gpt-4o", etc.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.model = os.getenv("LLM_MODEL", "llama3.2")
        self._client = None
        self._initialized = False

    def _init(self):
        if self._initialized:
            return

        if self.provider == "ollama":
            try:
                import ollama
                host = os.getenv("OLLAMA_HOST", "localhost")
                port = os.getenv("OLLAMA_PORT", "11434")
                self._ollama = ollama.Client(host=f"http://{host}:{port}")
                logger.info(f"LLM: Ollama at {host}:{port}, model={self.model}")
            except ImportError:
                raise RuntimeError("ollama package not installed. Run: pip install ollama")

        elif self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
                self._client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"LLM: Anthropic, model={self.model}")
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        elif self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY environment variable not set")
                self._client = openai.OpenAI(api_key=api_key)
                logger.info(f"LLM: OpenAI, model={self.model}")
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")

        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}. Use: ollama, anthropic, openai")

        self._initialized = True

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        """
        Synchronous LLM completion. Returns the response text.
        Falls back to empty string on error (agents handle degraded state gracefully).
        """
        self._init()

        try:
            if self.provider == "ollama":
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                response = self._ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"num_predict": max_tokens},
                )
                return response["message"]["content"].strip()

            elif self.provider == "anthropic":
                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system

                response = self._client.messages.create(**kwargs)
                return response.content[0].text.strip()

            elif self.provider == "openai":
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"LLM completion failed ({self.provider}): {e}")
            return ""

    def suggest_approach(self, task_description: str, past_failures: list[str]) -> dict:
        """
        Ask the LLM to suggest a task approach given past failures.
        Returns a structured approach dict, or empty dict on failure.
        """
        failure_text = "\n".join(f"- {f}" for f in past_failures) if past_failures else "None"

        system = (
            "You are an agent strategy advisor. Given a task description and past failures, "
            "suggest a concrete approach as a JSON object with keys: "
            "'method' (string), 'parameters' (dict), 'rationale' (string). "
            "Respond with ONLY valid JSON."
        )

        prompt = (
            f"Task: {task_description}\n\n"
            f"Past failures:\n{failure_text}\n\n"
            "Suggest a new approach:"
        )

        raw = self.complete(prompt, system=system, max_tokens=256)

        if not raw:
            return {}

        # Extract JSON from response
        import json
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"method": "llm_suggested", "rationale": raw}

    def is_available(self) -> bool:
        """Check if the LLM is reachable."""
        try:
            self._init()
            return True
        except Exception:
            return False
