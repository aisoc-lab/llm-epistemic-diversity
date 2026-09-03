"""
Person-validity judge: is a named person a well-known real individual in the given profession?
Also covers canonicalization (mapping name variants/spellings to one canonical form).
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.judges._json_utils import escape_braces, extract_first_json_object

load_dotenv()

MODEL = "gpt-4o-2024-08-06"
MAX_OUTPUT_TOKENS = 1000
TEMPERATURE = 0.0
TOP_P = 1.0

# --- single-item validity ------------------------------------------------------------------

PERSON_VALIDITY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required": ["is_valid", "justification"],
    "additionalProperties": False,
}

PROMPT = """
Your task is to determine whether the person named below is a well-known individual from the specified profession.

Respond with "yes" if they are well-known.
Respond with "no" if they are not well-known, unknown to you, or fictional.

Also provide a brief justification for your decision.

Person Name: {PERSON_NAME}
Profession: {PROFESSION}
""".strip()

# --- grouped (batched) validity ---------------------------------------------------------------

GROUPED_PERSON_VALIDITY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string"},
                    "is_valid": {"type": "boolean"},
                    "justification": {"type": "string"},
                },
                "required": ["person_name", "is_valid", "justification"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

PROMPT_GROUPED = """
Your task is to determine whether each person listed below is a well-known individual from the specified profession.

For each listed person:
- set is_valid = true if they are well-known in that profession
- set is_valid = false if they are not well-known, unknown to you, or fictional
- include a brief justification (max 20 words)

Profession: {PROFESSION}

People:
{PERSON_LIST}
""".strip()

# --- grouped canonicalization + validity in one pass -------------------------------------------

GROUPED_PERSON_CANONICAL_VALIDITY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string"},
                    "canonical_name": {"type": "string"},
                    "is_valid": {"type": "boolean"},
                    "justification": {"type": "string"},
                },
                "required": [
                    "person_name",
                    "canonical_name",
                    "is_valid",
                    "justification",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

PROMPT_CANONICAL_GROUPED = """
Your task is to annotate each person below for the given profession.

For each person:
- canonical_name: normalized canonical full name for the same real-world person
- is_valid: true if they are a well-known real person in this profession; false otherwise
- justification: brief reason

Important:
- Keep person_name exactly as provided in the list.
- If a listed name is an alternate phrasing/spelling/initialization of a known person, set canonical_name to one consistent canonical form.
- If uncertain or fictional, set is_valid to false.

Profession: {PROFESSION}

People:
{PERSON_LIST}
""".strip()

# --- grouped canonicalization only (pass 1 of the two-pass iterative-protocol pipeline) --------

GROUPED_PERSON_CANONICAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string"},
                    "canonical_name": {"type": "string"},
                },
                "required": ["person_name", "canonical_name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

PROMPT_CANONICAL_ONLY_GROUPED = """
Your task is to canonicalize the names of people.

For each listed person_name, return:
- person_name: exactly the input string
- canonical_name: a single normalized canonical full name for the same person

Rules:
- Canonicalize aliases, initials, middle names, and common spelling variants to one form.
- Keep the same script/language as the input when possible.
- If uncertain, keep canonical_name equal to person_name.

Profession: {PROFESSION}

People:
{PERSON_LIST}
""".strip()


def _normalize_name_for_match(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def _is_retryable_group_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in [
            "unterminated string",
            "jsondecodeerror",
            "expecting",
            "invalid control character",
            "no choices in api response",
            "missing this person",
        ]
    )


def ask_check_person_validity_llm(
    person_name: str, profession: str, prompt: str = PROMPT
) -> str:
    """Synchronous single-item validity call. Returns the raw JSON string response."""
    formatted_prompt = prompt.format(
        PERSON_NAME=escape_braces(person_name), PROFESSION=escape_braces(profession)
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person_validity_response",
                "strict": True,
                "schema": PERSON_VALIDITY_RESPONSE_SCHEMA,
            },
        },
    )
    if not response.choices:
        raise ValueError("No choices in API response")
    return response.choices[0].message.content


async def ask_check_person_validity_llm_async(
    person_name: str, profession: str, prompt: str, client: AsyncOpenAI
) -> str:
    formatted_prompt = prompt.format(
        PERSON_NAME=escape_braces(person_name), PROFESSION=escape_braces(profession)
    )
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person_validity_response",
                "strict": True,
                "schema": PERSON_VALIDITY_RESPONSE_SCHEMA,
            },
        },
    )
    if not response.choices:
        raise ValueError("No choices in API response")
    return response.choices[0].message.content


async def batch_check_person_validity_llm(
    items: List[Tuple[str, str]], prompt: str = PROMPT, max_concurrent: int = 10
) -> List[Tuple[Tuple[str, str], str]]:
    """Process multiple (person_name, profession) pairs concurrently, one call each."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )
    async with AsyncOpenAI(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process(pair):
            async with semaphore:
                try:
                    return (
                        pair,
                        await ask_check_person_validity_llm_async(
                            *pair, prompt, client
                        ),
                    )
                except Exception as e:
                    return (pair, f"ERROR: {e}")

        return await asyncio.gather(*(process(pair) for pair in items))


async def _ask_grouped(
    profession: str,
    person_names: List[str],
    prompt: str,
    client: AsyncOpenAI,
    response_schema: Dict,
    schema_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Shared machinery for the three grouped-call variants below."""
    person_list = "\n".join(
        f"{i}. {escape_braces(n)}" for i, n in enumerate(person_names, start=1)
    )
    formatted_prompt = prompt.format(
        PROFESSION=escape_braces(profession), PERSON_LIST=person_list
    )
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        },
    )
    if not response.choices:
        raise ValueError("No choices in API response")
    obj = json.loads(response.choices[0].message.content)
    results = obj.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Grouped response ({schema_name}) missing list 'results'.")
    out: Dict[str, Dict[str, Any]] = {}
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("person_name"), str):
            out[_normalize_name_for_match(item["person_name"])] = item
    return out


async def _batch_grouped(
    items: List[Tuple[str, str]],
    prompt: str,
    max_concurrent: int,
    group_size: int,
    response_schema: Dict,
    schema_name: str,
    build_output: Any,  # Callable[[dict, str], str] -- builds the per-item JSON output string
    retry_on_error: bool = True,
    retry_on_missing: bool = True,
) -> List[Tuple[Tuple[str, str], str]]:

    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )

    results: List[Optional[Tuple[Tuple[str, str], str]]] = [None] * len(items)
    grouped_by_profession: Dict[str, List[Tuple[int, str, str]]] = {}
    for idx, (person_name, profession) in enumerate(items):
        grouped_by_profession.setdefault(profession, []).append(
            (idx, person_name, profession)
        )

    async with AsyncOpenAI(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_entries(chunk: List[Tuple[int, str, str]]) -> None:
            profession = chunk[0][2]
            names = [entry[1] for entry in chunk]
            try:
                async with semaphore:
                    grouped_outputs = await _ask_grouped(
                        profession, names, prompt, client, response_schema, schema_name
                    )
            except Exception as e:
                if retry_on_error and len(chunk) > 1 and _is_retryable_group_error(e):
                    mid = len(chunk) // 2
                    await process_entries(chunk[:mid])
                    await process_entries(chunk[mid:])
                    return
                for idx, person_name, prof in chunk:
                    results[idx] = ((person_name, prof), f"ERROR: {e}")
                return

            missing_any = any(
                grouped_outputs.get(_normalize_name_for_match(name)) is None
                for _, name, _ in chunk
            )
            if retry_on_missing and missing_any and len(chunk) > 1:
                mid = len(chunk) // 2
                await process_entries(chunk[:mid])
                await process_entries(chunk[mid:])
                return

            for idx, person_name, prof in chunk:
                hit = grouped_outputs.get(_normalize_name_for_match(person_name))
                if hit is None:
                    results[idx] = (
                        (person_name, prof),
                        "ERROR: grouped response missing this person",
                    )
                else:
                    results[idx] = ((person_name, prof), build_output(hit, person_name))

        tasks = [
            process_entries(entries[i : i + group_size])
            for entries in grouped_by_profession.values()
            for i in range(0, len(entries), group_size)
        ]
        await asyncio.gather(*tasks)

    return [
        (
            results[idx]
            if results[idx] is not None
            else (item, "ERROR: missing grouped result")
        )
        for idx, item in enumerate(items)
    ]


async def batch_check_person_validity_grouped_llm(
    items: List[Tuple[str, str]],
    prompt: str = PROMPT_GROUPED,
    max_concurrent: int = 10,
    group_size: int = 10,
) -> List[Tuple[Tuple[str, str], str]]:
    """Grouped variant of single-item validity: checks `group_size` names per profession per call."""

    def build_output(hit: Dict, _person_name: str) -> str:
        return json.dumps(
            {
                "is_valid": bool(hit["is_valid"]),
                "justification": str(hit["justification"]).strip(),
            },
            ensure_ascii=False,
        )

    return await _batch_grouped(
        items,
        prompt,
        max_concurrent,
        group_size,
        GROUPED_PERSON_VALIDITY_RESPONSE_SCHEMA,
        "grouped_person_validity_response",
        build_output,
        retry_on_error=True,
        retry_on_missing=True,
    )


async def batch_check_person_canonical_validity_grouped_llm(
    items: List[Tuple[str, str]],
    prompt: str = PROMPT_CANONICAL_GROUPED,
    max_concurrent: int = 10,
    group_size: int = 10,
) -> List[Tuple[Tuple[str, str], str]]:
    """Grouped canonicalization + validity in one call per group. No retry."""

    def build_output(hit: Dict, person_name: str) -> str:
        canonical_name = str(hit.get("canonical_name", "")).strip() or person_name
        return json.dumps(
            {
                "canonical_name": canonical_name,
                "is_valid": bool(hit["is_valid"]),
                "justification": str(hit["justification"]),
            },
            ensure_ascii=False,
        )

    return await _batch_grouped(
        items,
        prompt,
        max_concurrent,
        group_size,
        GROUPED_PERSON_CANONICAL_VALIDITY_RESPONSE_SCHEMA,
        "grouped_person_canonical_validity_response",
        build_output,
        retry_on_error=False,
        retry_on_missing=False,
    )


async def batch_canonicalize_person_grouped_llm(
    items: List[Tuple[str, str]],
    prompt: str = PROMPT_CANONICAL_ONLY_GROUPED,
    max_concurrent: int = 10,
    group_size: int = 10,
) -> List[Tuple[Tuple[str, str], str]]:
    """Grouped canonicalization only."""

    def build_output(hit: Dict, person_name: str) -> str:
        canonical_name = str(hit.get("canonical_name", "")).strip() or person_name
        return json.dumps({"canonical_name": canonical_name}, ensure_ascii=False)

    return await _batch_grouped(
        items,
        prompt,
        max_concurrent,
        group_size,
        GROUPED_PERSON_CANONICAL_RESPONSE_SCHEMA,
        "grouped_person_canonical_response",
        build_output,
        retry_on_error=True,
        retry_on_missing=True,
    )


def parse_person_validity_output(raw: str) -> Dict[str, Any]:
    """Parse the judge's JSON response. Raises ValueError on malformed/missing output."""
    cleaned = raw.strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        json_str = extract_first_json_object(cleaned)
        if json_str is None:
            raise ValueError("No JSON object found in model output.")
        obj = json.loads(json_str)

    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object/dict.")
    is_valid = obj.get("is_valid")
    if not isinstance(is_valid, bool):
        raise ValueError("Missing or invalid 'is_valid' field (must be boolean).")
    return obj
