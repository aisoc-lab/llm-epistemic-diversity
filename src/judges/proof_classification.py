"""
Proof-strategy classification judge: which named proof strategy (if any) did a model use?
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.judges._json_utils import extract_first_json_object

load_dotenv()

MODEL = "gpt-5.2-2025-12-11"
MAX_OUTPUT_TOKENS = 1000
TEMPERATURE = 0.0
TOP_P = 1.0

PROOFCHECKER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["category", "rationale"],
    "additionalProperties": False,
}

PROMPT = """
You will be given the following information:
1. A mathematical theorem.
2. A list of some possible proofs in the attached PDF file. The theorem can be proved in many different ways. All the proofs in the attached file are correct. The file "titles" the solutions as "first solution", "second solution", etc.
3. A proof proposed by a respondent.

Your task is to classify the approach taken by the respondent. If the respondent took one of the approaches from the attached PDF file, provide the title of the approach. If the respondent took an approach not contained in the PDF file, provide a descriptive name of the proof based on its structure, e.g., "Proof Using Binomial Theorem".

Important to know:
1. We do not care whether the proof is mathematically correct. Focus only on the approach used.
2. If the proof tries many strategies but does not finish any of them, classify it as "invalid strategy" and explain why.

The theorem to be proved: {THEOREM}

The proof by the respondent is: {PROOF}
""".strip()


def upload_pdf_file(pdf_path: str, client: OpenAI) -> str:
    """Upload a reference-solutions PDF to OpenAI's Files API; returns the file ID."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    with open(path, "rb") as f:
        return client.files.create(file=f, purpose="user_data").id


async def upload_pdf_file_async(pdf_path: str, client: AsyncOpenAI) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    with open(path, "rb") as f:
        return (await client.files.create(file=f, purpose="user_data")).id


def _content_blocks(pdf_file_id: str, formatted_prompt: str) -> List[Dict]:
    return [
        {"type": "file", "file": {"file_id": pdf_file_id}},
        {"type": "text", "text": formatted_prompt},
    ]


def ask_proofchecker_llm(
    theorem: str, proof: str, pdf_file_id: str, prompt: str = PROMPT
) -> str:
    """Synchronous single-item classification call. Returns the raw JSON string response."""
    formatted_prompt = prompt.format(THEOREM=theorem, PROOF=proof)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": _content_blocks(pdf_file_id, formatted_prompt)}
        ],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "proofchecker_response",
                "strict": True,
                "schema": PROOFCHECKER_RESPONSE_SCHEMA,
            },
        },
    )
    return response.choices[0].message.content


async def ask_proofchecker_llm_async(
    theorem: str, proof: str, pdf_file_id: str, prompt: str, client: AsyncOpenAI
) -> str:
    formatted_prompt = prompt.format(THEOREM=theorem, PROOF=proof)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": _content_blocks(pdf_file_id, formatted_prompt)}
        ],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "proofchecker_response",
                "strict": True,
                "schema": PROOFCHECKER_RESPONSE_SCHEMA,
            },
        },
    )
    return response.choices[0].message.content


async def batch_proofchecker_llm(
    items: List[Tuple[str, str, str]], prompt: str = PROMPT, max_concurrent: int = 10
) -> List[Tuple[str, str, str, str]]:
    """
    Process multiple (theorem, proof, pdf_path) items concurrently. PDF uploads are cached per
    path so each of the 9 problems' PDFs is uploaded once regardless of how many proofs (model x
    sample) reference it.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file"
        )

    pdf_file_cache: Dict[str, str] = {}
    async with AsyncOpenAI(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process(theorem: str, proof: str, pdf_path: str):
            async with semaphore:
                try:
                    if pdf_path not in pdf_file_cache:
                        pdf_file_cache[pdf_path] = await upload_pdf_file_async(
                            pdf_path, client
                        )
                    output = await ask_proofchecker_llm_async(
                        theorem, proof, pdf_file_cache[pdf_path], prompt, client
                    )
                    return (theorem, proof, pdf_path, output)
                except Exception as e:
                    return (theorem, proof, pdf_path, f"ERROR: {e}")

        return await asyncio.gather(*(process(*item) for item in items))


def parse_proofchecker_output(raw: str) -> Dict[str, Any]:
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
    category = obj.get("category")
    if not isinstance(category, str):
        raise ValueError("Missing or invalid 'category' field.")
    obj["category"] = category.strip()
    obj["rationale"] = (
        obj.get("rationale", "").strip()
        if isinstance(obj.get("rationale"), str)
        else ""
    )
    return obj
