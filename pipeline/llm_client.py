"""
LLM Client (GitHub Models - OpenAI Compatible)

Uses GPT-4o via GitHub Models' free inference endpoint.
Handles rate limiting, retries, and structured JSON output.
"""
import json
import time
from openai import OpenAI

from pipeline.config import GITHUB_TOKEN, LLM_MODEL, LLM_MODEL_LIGHT, LLM_MODEL_REASON, LLM_BASE_URL


_client = None


def _get_client():
    global _client
    if _client is None:
        if not GITHUB_TOKEN:
            raise ValueError(
                "GITHUB_TOKEN not set. Create a .env file with:\n"
                "  GITHUB_TOKEN=your-github-pat-here\n"
                "Generate at: https://github.com/settings/tokens"
            )
        _client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=GITHUB_TOKEN,
        )
    return _client


def call_gemini(
    prompt: str,
    system_instruction: str | None = None,
    model: str = LLM_MODEL,
    temperature: float = 0.2,
    max_retries: int = 5,
    expect_json: bool = False,
) -> str:
    """
    Call GitHub Models API with retry logic.

    Note: Function is named call_gemini for backward compatibility
    with other pipeline modules. Actually uses GitHub Models.

    Args:
        prompt: The user prompt.
        system_instruction: System-level instruction for the model.
        model: Model name (default: gpt-4o via GitHub Models).
        temperature: Sampling temperature (lower = more deterministic).
        max_retries: Number of retries on failure.
        expect_json: If True, request structured JSON output.

    Returns:
        The model's text response.
    """
    client = _get_client()

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }

    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()

            if expect_json:
                json.loads(text)

            return text

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                "429" in str(e)
                or "rate limit" in err_str
                or "too many requests" in err_str
                or "quota" in err_str
            )
            if attempt < max_retries - 1:
                # Rate-limit errors need a much longer back-off
                wait = 60 * (attempt + 1) if is_rate_limit else 2 ** (attempt + 1)
                tag = "rate-limit" if is_rate_limit else "error"
                print(f"[LLM] Attempt {attempt + 1} failed ({tag}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM API failed after {max_retries} attempts: {e}")


def call_gemini_json(
    prompt: str,
    system_instruction: str | None = None,
    model: str = LLM_MODEL,
    temperature: float = 0.1,
) -> dict | list:
    """
    Call LLM and parse the response as JSON.

    Returns:
        Parsed JSON (dict or list).
    """
    text = call_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        model=model,
        temperature=temperature,
        expect_json=True,
    )
    return json.loads(text)
