"""Ollama LLM client for answer generation."""

import json
import logging
from collections.abc import AsyncGenerator

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama local LLM inference."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.OLLAMA_LLM_MODEL
        self.base_url = settings.ollama_base_url

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt (with context already formatted).
            system_prompt: System-level instructions for the model.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text response.
        """
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM.

        Args:
            prompt: The user prompt (with context already formatted).
            system_prompt: System-level instructions for the model.
            temperature: Sampling temperature.

        Yields:
            Individual text tokens as they are generated.
        """
        async with (
            httpx.AsyncClient(timeout=120) as client,
            client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "options": {"temperature": temperature},
                    "stream": True,
                },
            ) as response,
        ):
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
