"""
HuggingFace provider (Llama, Mistral, Ministral, Qwen), local inference.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.common import GenerationParameters, LLMResult, OutputMetadata

load_dotenv()

_model_cache: Dict[str, object] = {}
_tokenizer_cache: Dict[str, object] = {}


def is_huggingface_model(model_name: str) -> bool:
    """HuggingFace models are identified by an org/model slash, or a known family name."""
    return "/" in model_name or any(
        name in model_name.lower()
        for name in ["llama", "mistral", "phi", "gemma", "qwen", "falcon", "mpt"]
    )


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_and_tokenizer(model_name: str, device: Optional[torch.device] = None):
    """Load (and cache) a model + tokenizer. Ministral models get a dedicated loading path."""
    if device is None:
        device = get_device()
    if model_name in _model_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name]

    logging.info(f"Loading model {model_name} on {device}...")
    if "ministral" in model_name.lower():
        from transformers import (
            FineGrainedFP8Config,
            Mistral3ForConditionalGeneration,
            MistralCommonBackend,
        )

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=FineGrainedFP8Config(dequantize=True),
        )
        tokenizer = MistralCommonBackend.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type == "cpu":
            model = model.to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    _model_cache[model_name] = model
    _tokenizer_cache[model_name] = tokenizer
    return model, tokenizer


def _format_prompt(tokenizer, prompt: str, system_prompt: Optional[str] = None) -> str:
    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            logging.warning(
                f"Chat template failed for {tokenizer.__class__.__name__}: {e}. Falling back."
            )
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


def _extract_response(
    tokenizer, outputs, prompt_len: int, input_ids
) -> Tuple[str, str]:
    """Slice out newly-generated tokens; cross-check against a text-level diff for robustness."""
    new_token_ids = outputs[prompt_len:]
    response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    full_output_text = tokenizer.decode(outputs, skip_special_tokens=True)
    if full_output_text.startswith(input_text):
        cleaned = full_output_text[len(input_text) :].strip()
        if cleaned:
            input_end = input_text[-min(50, len(input_text)) :]
            if (
                not response_text.startswith(input_end)
                and len(cleaned) <= len(response_text) + 20
            ):
                response_text = cleaned

    generated_text = tokenizer.decode(outputs, skip_special_tokens=False)
    return response_text.strip(), generated_text


def query_model(
    prompt: str,
    model_name: str,
    max_output_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 1.0,
    system_prompt: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[LLMResult, Dict]:
    """Single-prompt generation."""
    if device is None:
        device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    is_ministral = "ministral" in model_name.lower()

    if is_ministral:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        )
        model_device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        formatted_prompt = None
    else:
        formatted_prompt = _format_prompt(tokenizer, prompt, system_prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        generate_kwargs = dict(
            max_new_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
        )
        if is_ministral:
            outputs = model.generate(**inputs, **generate_kwargs)
        else:
            outputs = model.generate(
                **inputs,
                **generate_kwargs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    response_time_ms = int((time.time() - start_time) * 1000)
    timestamp = int(time.time())

    prompt_len = inputs["input_ids"].shape[-1]
    response_text, generated_text = _extract_response(
        tokenizer, outputs[0], prompt_len, inputs["input_ids"][0]
    )
    output_tokens = int(outputs[0].shape[-1] - prompt_len)

    gen_params = GenerationParameters(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )
    output_metadata = OutputMetadata(
        input_tokens=prompt_len,
        output_tokens=output_tokens,
        finish_reason="stop",
        timestamp=timestamp,
        response_time_ms=response_time_ms,
    )
    result = LLMResult(
        query=prompt,
        model_version=model_name,
        response_text=response_text,
        gen_params=gen_params,
        output_metadata=output_metadata,
    )
    raw_response = {
        "model": model_name,
        "formatted_prompt": formatted_prompt or "Ministral chat template applied",
        "generated_text": generated_text,
        "response_text": response_text,
    }
    return result, raw_response


def query_model_batch(
    prompts: List[str],
    model_name: str,
    max_output_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 1.0,
    system_prompt: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> List[Tuple[LLMResult, Dict]]:
    """Batched variant: one sample per prompt. Ministral falls back to sequential (chat-template requirement)."""
    if device is None:
        device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    if "ministral" in model_name.lower():
        return [
            query_model(
                p,
                model_name,
                max_output_tokens,
                temperature,
                top_p,
                system_prompt,
                device,
            )
            for p in prompts
        ]

    formatted_prompts = [_format_prompt(tokenizer, p, system_prompt) for p in prompts]
    inputs = tokenizer(
        formatted_prompts, return_tensors="pt", padding=True, truncation=False
    ).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_time_ms = int((time.time() - start_time) * 1000)
    timestamp = int(time.time())

    input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()
    results: List[Tuple[LLMResult, Dict]] = []
    for i in range(len(prompts)):
        input_len = input_lengths[i]
        response_text, generated_text = _extract_response(
            tokenizer, outputs[i], input_len, inputs["input_ids"][i][:input_len]
        )
        output_tokens = int(outputs[i].shape[-1] - input_len)

        gen_params = GenerationParameters(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
        )
        output_metadata = OutputMetadata(
            input_tokens=input_len,
            output_tokens=output_tokens,
            finish_reason="stop",
            timestamp=timestamp,
            response_time_ms=response_time_ms,
        )
        result = LLMResult(
            query=prompts[i],
            model_version=model_name,
            response_text=response_text,
            gen_params=gen_params,
            output_metadata=output_metadata,
        )
        raw_response = {
            "model": model_name,
            "formatted_prompt": formatted_prompts[i],
            "generated_text": generated_text,
            "response_text": response_text,
        }
        results.append((result, raw_response))
    return results


def query_with_messages(
    messages: List[Dict[str, str]],
    *,
    model_name: str,
    original_question: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    device: Optional[torch.device] = None,
) -> Tuple[LLMResult, Dict]:
    """Multi-turn query given a full chat history. Used by the multi-turn protocol."""
    if device is None:
        device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    is_ministral = "ministral" in model_name.lower()

    if is_ministral:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        model_device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        formatted_prompt = None
    else:
        if not (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        ):
            raise ValueError(
                f"Model {model_name} has no chat template; required for multi-turn."
            )
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        generate_kwargs = dict(
            max_new_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
        )
        if is_ministral:
            outputs = model.generate(**inputs, **generate_kwargs)
        else:
            outputs = model.generate(
                **inputs,
                **generate_kwargs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    response_time_ms = int((time.time() - start_time) * 1000)
    timestamp = int(time.time())

    prompt_len = inputs["input_ids"].shape[-1]
    response_text, generated_text = _extract_response(
        tokenizer, outputs[0], prompt_len, inputs["input_ids"][0]
    )
    output_tokens = int(outputs[0].shape[-1] - prompt_len)

    system_prompt = (
        messages[0]["content"]
        if messages and messages[0].get("role") == "system"
        else None
    )
    gen_params = GenerationParameters(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )
    output_metadata = OutputMetadata(
        input_tokens=prompt_len,
        output_tokens=output_tokens,
        finish_reason="stop",
        timestamp=timestamp,
        response_time_ms=response_time_ms,
    )
    result = LLMResult(
        query=original_question,
        model_version=model_name,
        response_text=response_text,
        gen_params=gen_params,
        output_metadata=output_metadata,
    )
    raw_response = {
        "model": model_name,
        "formatted_prompt": formatted_prompt or "Ministral chat template applied",
        "generated_text": generated_text,
        "response_text": response_text,
    }
    return result, raw_response
