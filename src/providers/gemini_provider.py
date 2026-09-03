"""
Gemini provider (2.5-flash and 3-flash-preview), single/chat-turn generation.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

from scripts.common import GenerationParameters, LLMResult, OutputMetadata

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


def is_gemini_model(model_name: str) -> bool:
    return "gemini" in model_name.lower()


def _client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables or .env file"
        )
    return genai.Client(api_key=api_key)


def _extract_system_prompt(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content"), messages[1:]
    return None, messages


def _messages_to_gemini_contents(
    messages: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    contents: List[Dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            continue
        gemini_role = (
            "model" if role == "assistant" else "user" if role == "user" else None
        )
        if gemini_role is None:
            continue
        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    return contents


def _parse_response(response_json: Dict) -> Tuple[str, Optional[str]]:
    candidates = response_json.get("candidates")
    if not candidates:
        logging.warning(
            "Gemini response contains no candidates. Returning empty answer."
        )
        return "", None
    parts = candidates[0]["content"]["parts"]
    finish_reason = candidates[0]["finish_reason"]
    if not parts:
        logging.warning("Gemini response contains no parts. Returning empty answer.")
        return "", finish_reason
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    return text, finish_reason


def _token_counts(usage_metadata: Dict) -> Tuple[int, int]:
    completion_tokens = sum(
        (usage_metadata.get(key) or 0)
        for key in ["candidates_token_count", "thoughts_token_count"]
    )
    input_tokens = sum(
        (usage_metadata.get(key) or 0)
        for key in ["prompt_token_count", "tool_use_prompt_token_count"]
    )
    return input_tokens, completion_tokens


def query_model(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 1000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    system_prompt: Optional[str] = None,
    thinking_budget: int = 0,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> Tuple[LLMResult, Dict]:
    """Single-turn query."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return query_with_messages(
        messages,
        model=model,
        original_question=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        thinking_budget=thinking_budget,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )


def query_with_messages(
    messages: List[Dict[str, str]],
    *,
    model: str,
    original_question: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int] = None,
    thinking_budget: int = 0,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> Tuple[LLMResult, Dict]:
    """Multi-turn query given a full chat history. Used by the multi-turn protocol."""
    client = _client()
    system_prompt, chat_messages = _extract_system_prompt(messages)
    contents = _messages_to_gemini_contents(chat_messages)

    gen_config = GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        thinking_config=ThinkingConfig(
            include_thoughts=True, thinking_budget=thinking_budget
        ),
    )

    attempt = 0
    response_json = None
    response_time_ms = 0
    timestamp = 0
    while True:
        attempt += 1
        start_time = time.time()
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=gen_config
            )
            response_time_ms = int((time.time() - start_time) * 1000)
            timestamp = int(time.time())
            response_json = response.model_dump()
        except Exception as e:
            if attempt <= max_retries:
                logging.warning(
                    f"Gemini request failed on attempt {attempt}/{max_retries + 1}: {e}. Retrying."
                )
                time.sleep(retry_backoff_s * (2 ** (attempt - 1)))
                continue
            raise
        break

    response_text, finish_reason = _parse_response(response_json)
    input_tokens, completion_tokens = _token_counts(
        response_json.get("usage_metadata", {}) or {}
    )

    gen_params = GenerationParameters(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        system_prompt=system_prompt,
    )
    output_metadata = OutputMetadata(
        input_tokens=input_tokens,
        output_tokens=completion_tokens,
        finish_reason=finish_reason or "unknown",
        timestamp=timestamp,
        response_time_ms=response_time_ms,
    )
    result = LLMResult(
        query=original_question,
        model_version=model,
        response_text=response_text,
        gen_params=gen_params,
        output_metadata=output_metadata,
    )
    return result, response_json
