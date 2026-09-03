"""
Person-extraction judge: identifies the primary named person a response is "about".
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.judges._json_utils import escape_braces, extract_first_json_object

load_dotenv()

MODEL = "gpt-4o-2024-08-06"
MAX_OUTPUT_TOKENS = 500
TEMPERATURE = 0.0
TOP_P = 1.0

PERSON_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"person_name": {"type": "string"}},
    "required": ["person_name"],
    "additionalProperties": False,
}

PROMPT = """
You are an expert in analyzing text to identify the main person being discussed.

Your task is to analyze a question and its corresponding answer to identify the name of the person that the answer is mainly about.

Guidelines:
- If the answer discusses a specific person, extract their full name (e.g., "Marie Curie", "Ludwig van Beethoven", "Alan Turing")
- If multiple people are mentioned, identify the primary person that the answer focuses on
- If no specific person is identified, use "none" as the person_name

Question:
{QUESTION}

Answer:
{ANSWER}
""".strip()

MULTI_PERSON_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "response_index": {"type": "integer"},
                    "person_name": {"type": "string"},
                },
                "required": ["response_index", "person_name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

MULTI_PROMPT = """
You are an expert in analyzing text. The answer below contains several distinct responses to
the same question (e.g., numbered or separated sub-answers).

For each distinct response, identify the name of the person it is mainly about (or "none" if no
specific person is identified). Number responses starting at 0, in the order they appear.

Question:
{QUESTION}

Answer (may contain multiple distinct responses):
{ANSWER}
""".strip()


def ask_extract_person_llm(question: str, answer: str, prompt: str = PROMPT) -> str:
    """Synchronous single-item extraction call. Returns the raw JSON string response."""
    formatted_prompt = prompt.format(
        QUESTION=escape_braces(question), ANSWER=escape_braces(answer)
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
                "name": "person_extraction_response",
                "strict": True,
                "schema": PERSON_EXTRACTION_RESPONSE_SCHEMA,
            },
        },
    )
    if not response.choices:
        raise ValueError("No choices in API response")
    return response.choices[0].message.content


async def ask_extract_person_llm_async(
    question: str, answer: str, prompt: str, client: AsyncOpenAI
) -> str:
    formatted_prompt = prompt.format(
        QUESTION=escape_braces(question), ANSWER=escape_braces(answer)
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
                "name": "person_extraction_response",
                "strict": True,
                "schema": PERSON_EXTRACTION_RESPONSE_SCHEMA,
            },
        },
    )
    if not response.choices:
        raise ValueError("No choices in API response")
    return response.choices[0].message.content


async def batch_extract_person_llm(
    items: List[Tuple[str, str]], prompt: str = PROMPT, max_concurrent: int = 10
) -> List[Tuple[Tuple[str, str], str]]:
    """Process multiple (question, answer) pairs concurrently, bounded by `max_concurrent`."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )

    async with AsyncOpenAI(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process(qa_pair):
            async with semaphore:
                try:
                    return (
                        qa_pair,
                        await ask_extract_person_llm_async(*qa_pair, prompt, client),
                    )
                except Exception as e:
                    return (qa_pair, f"ERROR: {e}")

        return await asyncio.gather(*(process(pair) for pair in items))


def parse_person_extraction_output(raw: str) -> Dict[str, Any]:
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
    person_name = obj.get("person_name")
    if not isinstance(person_name, str):
        raise ValueError("Missing or invalid 'person_name' field.")
    obj["person_name"] = person_name.strip()
    return obj


# --- multi-response variant (multi-output protocol) -------------------------------------------


async def ask_extract_multi_person_llm_async(
    question: str, answer: str, prompt: str, client: AsyncOpenAI
) -> List[Dict[str, Any]]:
    """Extract one person per sub-response from a multi-response completion."""
    formatted_prompt = prompt.format(
        QUESTION=escape_braces(question), ANSWER=escape_braces(answer)
    )
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_completion_tokens=MAX_OUTPUT_TOKENS * 4,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "multi_person_extraction_response",
                "strict": True,
                "schema": MULTI_PERSON_EXTRACTION_RESPONSE_SCHEMA,
            },
        },
    )
    if not response.choices:
        raise ValueError("No choices in API response")
    obj = json.loads(response.choices[0].message.content)
    results = obj.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Multi-person extraction response missing list 'results'.")
    return results


async def batch_extract_multi_person_llm(
    items: List[Tuple[str, str]], prompt: str = MULTI_PROMPT, max_concurrent: int = 10
) -> List[Tuple[Tuple[str, str], List[Dict[str, Any]]]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )

    async with AsyncOpenAI(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process(qa_pair):
            async with semaphore:
                try:
                    return (
                        qa_pair,
                        await ask_extract_multi_person_llm_async(
                            *qa_pair, prompt, client
                        ),
                    )
                except Exception as e:
                    return (
                        qa_pair,
                        [{"response_index": -1, "person_name": f"ERROR: {e}"}],
                    )

        return await asyncio.gather(*(process(pair) for pair in items))
