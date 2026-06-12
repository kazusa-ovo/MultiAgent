from typing import AsyncIterator,Any
from src.llm.providers import BaseProvider,OpenAIProvider

PROVIDER_ENV_MAP:dict[str, tuple[str,str]] = {
    "openai":("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "deepseek":("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
}

class LLMClient:
    """Unified async interface to LLM providers."""

    def __init__(
            self,
            provider: str = "openai",
            model:str = "gpt-4o",
            temperature: float = 0.7,
            max_tokens: int = 4096,
            api_key: str = "",
            base_url: str = "",
          ):
        if not api_key:
            api_key, base_url = self._read_env(provider)

        self._provider:BaseProvider = OpenAIProvider(model, api_key, base_url)
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens

    @staticmethod
    def _read_env(provider:str) -> tuple[str,str]:
        import os
        import dotenv
        dotenv.load_dotenv()

        key_env,url_env = PROVIDER_ENV_MAP.get(provider,("",""))
        return os.environ.get(key_env,""),os.environ.get(url_env,"")

    @property
    def model(self) -> str:
        return self._provider.model

    async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            ) -> dict[str,Any]:
        return await self._provider.chat(
            messages = messages,
            tools = tools,
            temperature = temperature if temperature is not None else self._default_temperature,
            max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    )-> AsyncIterator[str]:
        async for chunk in self._provider.chat_stream(
            messages = messages,
            tools = tools,
            temperature = temperature if temperature is not None else self._default_temperature,
            max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens,
        ):
            yield chunk
