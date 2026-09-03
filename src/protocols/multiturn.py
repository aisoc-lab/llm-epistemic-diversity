"""
Multi-turn diversity protocol (follow-up requests within one conversation)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import tqdm

from scripts.common import PROMPT_COLUMN, get_datasets, get_output_file
from src.providers import (
    get_temperature_for_model,
    get_top_p_for_model,
    query_model_with_messages,
)

PROTOCOL_NAME = "multiturn"
TEST_RUN_NUM_QUERIES = 2
TURN_RECORD_TYPE = "multiturn_person_diversity_turn_v1"
DEFAULT_FOLLOWUP_MESSAGE = "Can you suggest a different person?"


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


def rebuild_messages_from_turns(
    turn_records: List[Dict[str, Any]], system_prompt: Optional[str]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for rec in turn_records:
        user_message, response_text = rec.get("user_message"), rec.get("response_text")
        if not isinstance(user_message, str) or not isinstance(response_text, str):
            continue
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": response_text})
    return messages


def reconstruct_state_from_records(
    records: List[Dict[str, Any]], n_turns: int, system_prompt: Optional[str] = None
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Rebuild {(query_id, sample_id): {completed_turn_ids, messages}} for resume."""
    per_conv: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rec in records:
        if (
            rec.get("record_type") != TURN_RECORD_TYPE
            or rec.get("event_status") != "ok"
        ):
            continue
        qid, sid, tid = rec.get("query_id"), rec.get("sample_id"), rec.get("turn_id")
        if not all(isinstance(x, int) for x in (qid, sid, tid)) or not (
            0 <= tid < n_turns
        ):
            continue
        per_conv.setdefault((qid, sid), []).append(rec)

    state: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, recs in per_conv.items():
        recs_sorted = sorted(recs, key=lambda r: r.get("turn_id", 0))
        completed: Set[int] = set()
        sequential: List[Dict[str, Any]] = []
        for rec in recs_sorted:
            if rec["turn_id"] != len(sequential):
                break  # gap: stop reconstructing past it
            completed.add(rec["turn_id"])
            sequential.append(rec)
        state[key] = {
            "completed_turn_ids": completed,
            "messages": rebuild_messages_from_turns(
                sequential, system_prompt=system_prompt
            ),
        }
    return state


def get_next_missing_turn_id(completed: Set[int], n_turns: int) -> Optional[int]:
    for tid in range(n_turns):
        if tid not in completed:
            return tid
    return None


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


def run_multiturn(
    models: List[str],
    datasets_filter: Optional[List[str]],
    n_turns: int,
    n_samples: int,
    followup_message: str,
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
    total_work = sum(
        n * n_samples * n_turns * len(models) for n in n_per_dataset.values()
    )
    bar = tqdm.tqdm(
        total=total_work, desc="Multi-turn: conversation turns", unit="turn"
    )

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
            state_by_conv = reconstruct_state_from_records(
                existing_records, n_turns, system_prompt=system_prompt
            )
            selected_keys = {
                (r["query_id"], sid) for r in rows for sid in range(n_samples)
            }
            precompleted = sum(
                len(s.get("completed_turn_ids", set()))
                for k, s in state_by_conv.items()
                if k in selected_keys
            )
            if precompleted:
                bar.update(precompleted)

            for row in rows:
                query_id, base_question = row["query_id"], row["base_question"]
                profession, prompt_key = row["profession"], row["prompt_key"]

                for sample_id in range(n_samples):
                    conv_key = (query_id, sample_id)
                    conv_state = state_by_conv.setdefault(
                        conv_key, {"completed_turn_ids": set(), "messages": []}
                    )
                    completed, messages = conv_state["completed_turn_ids"], list(
                        conv_state["messages"]
                    )

                    while True:
                        turn_id = get_next_missing_turn_id(completed, n_turns)
                        if turn_id is None:
                            break

                        if turn_id == 0:
                            if not messages:
                                messages = []
                                if system_prompt:
                                    messages.append(
                                        {"role": "system", "content": system_prompt}
                                    )
                                messages.append(
                                    {"role": "user", "content": base_question}
                                )
                            user_message, turn_followup = base_question, None
                        else:
                            user_message, turn_followup = (
                                followup_message,
                                followup_message,
                            )
                            messages.append(
                                {"role": "user", "content": followup_message}
                            )

                        event = {
                            "record_type": TURN_RECORD_TYPE,
                            "timestamp": int(time.time()),
                            "run_id": run_id,
                            "model": model_name,
                            "dataset": dataset_name,
                            "query_id": query_id,
                            "prompt_key": prompt_key,
                            "profession": profession,
                            "base_question": base_question,
                            "sample_id": sample_id,
                            "turn_id": turn_id,
                            "user_message": user_message,
                            "followup_message": turn_followup,
                            "response_text": "",
                            "gen_params": {},
                            "output_metadata": {},
                            "event_status": "error",
                            "error_message": "",
                        }
                        try:
                            model_temp = get_temperature_for_model(
                                model_name, temperature
                            )
                            model_top_p = get_top_p_for_model(model_name, top_p)
                            result, _ = query_model_with_messages(
                                messages=messages,
                                model_name=model_name,
                                original_question=base_question,
                                max_output_tokens=max_output_tokens,
                                temperature=model_temp,
                                top_p=model_top_p,
                                **llama_kwargs,
                            )
                            event["response_text"] = result.response_text
                            event["gen_params"] = asdict(result.gen_params)
                            event["output_metadata"] = asdict(result.output_metadata)
                            event["event_status"] = "ok"
                            messages.append(
                                {"role": "assistant", "content": result.response_text}
                            )
                            completed.add(turn_id)
                            conv_state["messages"], conv_state["completed_turn_ids"] = (
                                messages,
                                completed,
                            )
                        except Exception as e:
                            event["error_message"] = str(e)
                            append_jsonl_record(output_file, event)
                            bar.update(1)
                            logging.error(
                                f"Failed turn {turn_id} for {model_name}/{dataset_name}/{query_id}/{sample_id}: {e}"
                            )
                            break

                        append_jsonl_record(output_file, event)
                        bar.update(1)

    max_workers = model_workers or len(models)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_model, m) for m in models]
        for _ in as_completed(futures):
            pass
    bar.close()
    logging.info("Completed multi-turn-diversity generation.")
