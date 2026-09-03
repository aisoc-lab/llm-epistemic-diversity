"""
WildChat regime-classification judge: for each query, how many valid answers does it admit, and is it correctly specified or underspecified?
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.judges._json_utils import extract_first_json_object

load_dotenv()

MODEL = "gpt-5.2-2025-12-11"
MAX_OUTPUT_TOKENS = 500
TEMPERATURE = 0.0
TOP_P = 1.0

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "number_of_responses": {
            "type": "string",
            "enum": ["one", "multiple", "infinite"],
        },
        "number_of_responses_rationale": {"type": "string"},
        "specification": {
            "type": "string",
            "enum": ["correctly_specified", "underspecified"],
        },
        "specification_rationale": {"type": "string"},
    },
    "required": [
        "number_of_responses",
        "number_of_responses_rationale",
        "specification",
        "specification_rationale",
    ],
    "additionalProperties": False,
}

PROMPT = """
You are an expert in analyzing open-ended user queries. Such queries sometimes allow for multiple valid responses.
Multiple valid responses means that when asked the same query, different people could provide different answers. A single valid response means that all surveyed participants would likely provide the same answer.

Your task is to analyze each query and determine:
1. How many valid answers it allows. The options are:
    - "one": The query allows for a single valid answer.
    - "infinite": There are infinitely many valid answers.
    - "multiple": More than one but less than infinitely many valid answers.

2. Whether the query is well-specified or underspecified. The options are:
    - "correctly_specified": The query is clear enough that a concrete answer can be given without guessing the user's intent.
    - "underspecified": The query lacks the detail required to give a concrete answer.

You must also provide a clear rationale for each decision.

Query to analyze:
{QUESTION}
""".strip()


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )
    return OpenAI(api_key=api_key)


def ask_judge_llm(question: str, prompt: str = PROMPT) -> str:
    """Synchronous single-item classification call. Returns the raw JSON string response."""
    formatted_prompt = prompt.format(QUESTION=question)
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_response",
                "strict": True,
                "schema": JUDGE_RESPONSE_SCHEMA,
            },
        },
    )
    return response.choices[0].message.content


async def ask_judge_llm_async(question: str, prompt: str, client: AsyncOpenAI) -> str:
    formatted_prompt = prompt.format(QUESTION=question)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_response",
                "strict": True,
                "schema": JUDGE_RESPONSE_SCHEMA,
            },
        },
    )
    return response.choices[0].message.content


async def batch_judge_llm(
    items: List[str], prompt: str = PROMPT, max_concurrent: int = 10
) -> List[Tuple[str, str]]:
    """Process multiple queries concurrently, bounded by `max_concurrent`."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )
    async with AsyncOpenAI(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process(query: str):
            async with semaphore:
                try:
                    return (query, await ask_judge_llm_async(query, prompt, client))
                except Exception as e:
                    return (query, f"ERROR: {e}")

        return await asyncio.gather(*(process(q) for q in items))


def parse_judge_output(raw: str) -> Dict[str, Any]:
    """Parse the judge's JSON response. Raises ValueError on malformed/missing/out-of-enum output."""
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

    number_of_responses = obj.get("number_of_responses")
    if not isinstance(number_of_responses, str):
        raise ValueError("Missing or invalid 'number_of_responses' field.")
    obj["number_of_responses"] = number_of_responses.strip().lower()
    if obj["number_of_responses"] not in {"one", "multiple", "infinite"}:
        raise ValueError(
            f"Unexpected number_of_responses value: {obj['number_of_responses']}"
        )

    specification = obj.get("specification")
    if not isinstance(specification, str):
        raise ValueError("Missing or invalid 'specification' field.")
    obj["specification"] = specification.strip().lower()
    if obj["specification"] not in {"correctly_specified", "underspecified"}:
        raise ValueError(f"Unexpected specification value: {obj['specification']}")

    for key in ("number_of_responses_rationale", "specification_rationale"):
        obj[key] = obj[key].strip() if isinstance(obj.get(key), str) else ""

    return obj
