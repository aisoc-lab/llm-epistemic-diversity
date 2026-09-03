"""
Model-name-based provider dispatch.
"""

from typing import Dict, List, Optional, Tuple

from scripts.common import LLMResult
from src.providers.gemini_provider import is_gemini_model
from src.providers.gemini_provider import query_model as _query_gemini
from src.providers.gemini_provider import query_with_messages as _query_gemini_messages
from src.providers.huggingface_provider import is_huggingface_model
from src.providers.huggingface_provider import query_model as _query_hf
from src.providers.huggingface_provider import (
    query_model_batch as query_huggingface_batch,
)
from src.providers.huggingface_provider import query_with_messages as _query_hf_messages
from src.providers.openai_provider import is_gpt4o_model, is_gpt52_model
from src.providers.openai_provider import is_openai_model
from src.providers.openai_provider import query_model as _query_openai
from src.providers.openai_provider import query_with_messages as _query_openai_messages

__all__ = [
    "is_gpt4o_model",
    "is_gpt52_model",
    "is_gemini_model",
    "is_huggingface_model",
    "is_openai_model",
    "query_model_with_params",
    "query_model_with_messages",
    "query_huggingface_batch",
    "get_temperature_for_model",
    "get_top_p_for_model",
]

# Per-model temperature/top_p overrides (inactive by default)
MODEL_DEFAULT_TEMPERATURES: Dict[str, float] = {}
MODEL_DEFAULT_TOP_P: Dict[str, float] = {}


def get_temperature_for_model(model_name: str, default_temperature: float) -> float:
    if model_name in MODEL_DEFAULT_TEMPERATURES:
        return MODEL_DEFAULT_TEMPERATURES[model_name]
    for pattern, temp in MODEL_DEFAULT_TEMPERATURES.items():
        if pattern in model_name:
            return temp
    return default_temperature


def get_top_p_for_model(model_name: str, default_top_p: float) -> float:
    if model_name in MODEL_DEFAULT_TOP_P:
        return MODEL_DEFAULT_TOP_P[model_name]
    for pattern, top_p in MODEL_DEFAULT_TOP_P.items():
        if pattern in model_name:
            return top_p
    return default_top_p


def query_model_with_params(
    prompt: str,
    model_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> Tuple[LLMResult, Dict]:
    """Single-turn query, routed to the right provider by model name."""
    if is_gpt4o_model(model_name):
        return _query_openai(
            prompt, model_name, max_output_tokens, temperature, top_p, system_prompt
        )
    if is_gpt52_model(model_name):
        return _query_openai(
            prompt, model_name, max_output_tokens, temperature, top_p, system_prompt
        )
    if is_gemini_model(model_name):
        return _query_gemini(
            prompt,
            model_name,
            max_output_tokens,
            temperature,
            top_p,
            system_prompt=system_prompt,
        )
    if is_huggingface_model(model_name):
        return _query_hf(
            prompt,
            model_name,
            max_output_tokens,
            temperature,
            top_p,
            system_prompt,
            **kwargs,
        )
    raise ValueError(f"Unknown model type: {model_name}")


def query_model_with_messages(
    messages: List[Dict[str, str]],
    *,
    model_name: str,
    original_question: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    **kwargs,
) -> Tuple[LLMResult, Dict]:
    """Multi-turn query given a full chat history, routed by model name. Used by the multi-turn protocol."""
    if is_openai_model(model_name):
        return _query_openai_messages(
            messages,
            model=model_name,
            original_question=original_question,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    if is_gemini_model(model_name):
        return _query_gemini_messages(
            messages,
            model=model_name,
            original_question=original_question,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=kwargs.get("top_k"),
            thinking_budget=kwargs.get("thinking_budget", 0),
            max_retries=kwargs.get("max_retries", 2),
            retry_backoff_s=kwargs.get("retry_backoff_s", 1.0),
        )
    if is_huggingface_model(model_name):
        return _query_hf_messages(
            messages,
            model_name=model_name,
            original_question=original_question,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            device=kwargs.get("device"),
        )
    raise ValueError(f"Unknown model type for multi-turn querying: {model_name}")
