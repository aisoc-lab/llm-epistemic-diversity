"""
Multi-output diversity protocol (one call, N=10 requested responses)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import tqdm

from scripts.common import PROMPT_COLUMN, get_datasets, get_output_file
from src.providers import (
    get_temperature_for_model,
    get_top_p_for_model,
    query_model_with_params,
)

PROTOCOL_NAME = "multi_output"
TEST_RUN_NUM_QUERIES = 2
RECORD_TYPE = "multi_response_prompt_v1"
DEFAULT_RESPONSE_SUFFIX_TEMPLATE = "Generate {num_responses} different responses."


def build_effective_prompt(
    base_question: str,
    num_responses: int,
    suffix_template: str = DEFAULT_RESPONSE_SUFFIX_TEMPLATE,
) -> str:
    return f"{base_question.strip()}. {suffix_template.format(num_responses=num_responses)}"


def append_jsonl_record(output_file: Path, record: Dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_records(output_file: Path) -> List[Dict[str, Any]]:
    if not output_file.exists():
        return []
    records = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def get_completed_query_ids(
    records: List[Dict[str, Any]], selected_query_ids: Optional[Set[int]] = None
) -> Set[int]:
    completed = {
        rec["query_id"]
        for rec in records
        if rec.get("record_type") == RECORD_TYPE
        and rec.get("event_status") == "ok"
        and isinstance(rec.get("query_id"), int)
    }
    if selected_query_ids is not None:
        completed &= selected_query_ids
    return completed


def _build_rows(dataset_df, test_run: bool) -> List[Dict]:
    if test_run:
        dataset_df = dataset_df.iloc[:TEST_RUN_NUM_QUERIES]
    rows = []
    for i, row in dataset_df.iterrows():
        question = row.get(PROMPT_COLUMN)
        if not isinstance(question, str) or not question.strip():
            continue
        rows.append(
            {
                "query_id": int(i),
                "base_question": question.strip(),
                "profession": (
                    row.get("profession")
                    if isinstance(row.get("profession"), str)
                    else ""
                ),
                "prompt_key": (
                    row.get("prompt_key")
                    if isinstance(row.get("prompt_key"), str)
                    else ""
                ),
            }
        )
    return rows


def run_multi_output(
    models: List[str],
    datasets_filter: Optional[List[str]],
    num_responses: int,
    response_suffix_template: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str],
    clear_results: bool,
    test_run: bool,
    model_workers: Optional[int],
    device: Optional[str],
    output_suffix: Optional[str] = None,
) -> None:
    datasets = get_datasets()
    if datasets_filter is not None:
        selected = set(datasets_filter)
        datasets = [
            {stem: df} for d in datasets for stem, df in d.items() if stem in selected
        ]
    if not datasets:
        logging.error("No datasets found. Check data/datasets/.")
        return

    run_id = output_suffix or time.strftime("%Y%m%d_%H%M%S")
    n_per_dataset = {}
    for dataset in datasets:
        name, df = list(dataset.items())[0]
        n = len(df[PROMPT_COLUMN])
        n_per_dataset[name] = min(TEST_RUN_NUM_QUERIES, n) if test_run else n
    total_work = sum(n * len(models) for n in n_per_dataset.values())
    bar = tqdm.tqdm(total=total_work, desc="Multi-output: queries", unit="query")

    llama_kwargs = {}
    if device:
        import torch

        llama_kwargs["device"] = torch.device(device)

    def process_one_model(model_name: str) -> None:
        for dataset in datasets:
            dataset_name, df = list(dataset.items())[0]
            output_file = get_output_file(
                model_name,
                dataset_name,
                PROTOCOL_NAME,
                create_dir=True,
                delete_existing_file=clear_results,
                output_suffix=run_id,
            )
            rows = _build_rows(df, test_run=test_run)
            if not rows:
                continue

            existing_records = load_jsonl_records(output_file)
            selected_query_ids = {r["query_id"] for r in rows}
            completed_query_ids = get_completed_query_ids(
                existing_records, selected_query_ids
            )
            if completed_query_ids:
                bar.update(len(completed_query_ids))

            for row in rows:
                query_id = row["query_id"]
                if query_id in completed_query_ids:
                    continue
                base_question, profession, prompt_key = (
                    row["base_question"],
                    row["profession"],
                    row["prompt_key"],
                )
                effective_prompt = build_effective_prompt(
                    base_question, num_responses, response_suffix_template
                )

                event = {
                    "record_type": RECORD_TYPE,
                    "timestamp": int(time.time()),
                    "run_id": run_id,
                    "model": model_name,
                    "dataset": dataset_name,
                    "query_id": query_id,
                    "prompt_key": prompt_key,
                    "profession": profession,
                    "base_question": base_question,
                    "num_responses": num_responses,
                    "response_suffix_template": response_suffix_template,
                    "effective_prompt": effective_prompt,
                    "response_text": "",
                    "gen_params": {},
                    "output_metadata": {},
                    "event_status": "error",
                    "error_message": "",
                }
                try:
                    model_temp = get_temperature_for_model(model_name, temperature)
                    model_top_p = get_top_p_for_model(model_name, top_p)
                    result, _ = query_model_with_params(
                        prompt=effective_prompt,
                        model_name=model_name,
                        max_output_tokens=max_output_tokens,
                        temperature=model_temp,
                        top_p=model_top_p,
                        system_prompt=system_prompt,
                        **llama_kwargs,
                    )
                    event["response_text"] = result.response_text
                    event["gen_params"] = asdict(result.gen_params)
                    event["output_metadata"] = asdict(result.output_metadata)
                    event["event_status"] = "ok"
                except Exception as e:
                    event["error_message"] = str(e)
                    logging.error(
                        f"Failed query for {model_name}/{dataset_name}/{query_id}: {e}"
                    )

                append_jsonl_record(output_file, event)
                bar.update(1)

    max_workers = model_workers or len(models)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_model, m) for m in models]
        for _ in as_completed(futures):
            pass
    bar.close()
    logging.info("Completed multi-output-diversity generation.")
