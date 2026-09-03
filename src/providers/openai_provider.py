"""
OpenAI provider (GPT-4o and GPT-5.2), single/chat-turn generation.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from scripts.common import GenerationParameters, LLMResult, OutputMetadata

load_dotenv()

DEFAULT_MODEL = "gpt-4o-2024-08-06"


def is_gpt52_model(model_name: str) -> bool:
    return model_name.startswith("gpt-5.2")


def is_gpt4o_model(model_name: str) -> bool:
    return model_name.startswith("gpt-4o") or model_name.startswith("gpt-4")


def is_openai_model(model_name: str) -> bool:
    return (
        is_gpt4o_model(model_name)
        or is_gpt52_model(model_name)
        or model_name.startswith("gpt-")
    )


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )
    return OpenAI(api_key=api_key)


def _extract_system_prompt(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content"), messages[1:]
    return None, messages


def query_model(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 1000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    system_prompt: Optional[str] = None,
) -> Tuple[LLMResult, Dict]:
    """Single-turn query."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    result, response_dict = query_with_messages(
        messages,
        model=model,
        original_question=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return result, response_dict


def query_with_messages(
    messages: List[Dict[str, str]],
    *,
    model: str,
    original_question: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[LLMResult, Dict]:
    """Multi-turn query given a full chat history. Used by the multi-turn protocol."""
    client = _client()
    start_time = time.time()

    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if is_gpt52_model(model):
        request_kwargs["max_completion_tokens"] = max_output_tokens
    else:
        request_kwargs["max_tokens"] = max_output_tokens

    response = client.chat.completions.create(**request_kwargs)
    response_time_ms = int((time.time() - start_time) * 1000)
    response_dict = response.model_dump()

    response_text = response.choices[0].message.content or ""
    usage = response_dict.get("usage", {}) or {}
    system_prompt, _ = _extract_system_prompt(messages)

    gen_params = GenerationParameters(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )
    output_metadata = OutputMetadata(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        finish_reason=response.choices[0].finish_reason or "unknown",
        timestamp=int(time.time()),
        response_time_ms=response_time_ms,
    )
    result = LLMResult(
        query=original_question,
        model_version=model,
        response_text=response_text,
        gen_params=gen_params,
        output_metadata=output_metadata,
    )
    return result, response_dict
