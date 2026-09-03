"""
Latent diversity protocol (terative exclusion-based probing)

This protocol is generate-and-judge-in-the-loop: the exclusion list for round t+1 depends on
round t's judge output.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import tqdm

from scripts.common import PROMPT_COLUMN, get_datasets, get_output_file
from src.judges.person_extraction import (
    ask_extract_person_llm,
    parse_person_extraction_output,
)
from src.judges.person_validity import (
    ask_check_person_validity_llm,
    parse_person_validity_output,
)
from src.providers import (
    get_temperature_for_model,
    get_top_p_for_model,
    query_model_with_params,
)

PROTOCOL_NAME = "latent"
TEST_RUN_NUM_QUERIES = 2
ROUND_RECORD_TYPE = "iterative_person_diversity_round_v1"

INVALID_PERSON_NAMES = {
    "",
    "none",
    "unknown",
    "error",
    "parse_error",
    "nan",
    "n/a",
    "null",
}


def normalize_person_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip())


def is_valid_person_name(name: str) -> bool:
    return normalize_person_name(name).lower() not in INVALID_PERSON_NAMES


def add_person_case_insensitive(excluded_people: List[str], person_name: str) -> bool:
    """Add person_name if not already present (case-insensitive). Returns True if added."""
    norm = normalize_person_name(person_name)
    if not is_valid_person_name(norm):
        return False
    if norm.lower() in {p.lower() for p in excluded_people}:
        return False
    excluded_people.append(norm)
    return True


def build_iterative_prompt(
    base_question: str, profession: Optional[str], excluded_people: List[str]
) -> str:
    """Append the exclusion clause once at least one valid person has been named."""
    if not excluded_people:
        return base_question
    excluded = ", ".join(excluded_people)
    return f"{base_question}\n\nMake sure it is not about any of the following people: {excluded}."


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


def reconstruct_state_from_records(
    records: List[Dict[str, Any]], n_rounds: int
) -> Dict[int, Dict[str, Any]]:
    """Rebuild {query_id: {completed_round_ids, excluded_people}} from prior successful rounds, for resume."""
    per_query: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        if (
            rec.get("record_type") != ROUND_RECORD_TYPE
            or rec.get("event_status") != "ok"
        ):
            continue
        qid, rid = rec.get("query_id"), rec.get("round_id")
        if (
            not isinstance(qid, int)
            or not isinstance(rid, int)
            or not (0 <= rid < n_rounds)
        ):
            continue
        per_query.setdefault(qid, []).append(rec)

    state: Dict[int, Dict[str, Any]] = {}
    for qid, recs in per_query.items():
        recs_sorted = sorted(recs, key=lambda r: r.get("round_id", 0))
        completed: Set[int] = set()
        excluded: List[str] = []
        for rec in recs_sorted:
            completed.add(rec["round_id"])
            person_name = rec.get("person_name", "")
            added = rec.get("added_person_to_exclusion")
            if isinstance(added, bool):
                if added and isinstance(person_name, str):
                    add_person_case_insensitive(excluded, person_name)
            elif isinstance(person_name, str):
                add_person_case_insensitive(excluded, person_name)
        state[qid] = {"completed_round_ids": completed, "excluded_people": excluded}
    return state


def get_next_missing_round_id(completed: Set[int], n_rounds: int) -> Optional[int]:
    for rid in range(n_rounds):
        if rid not in completed:
            return rid
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


def run_latent(
    models: List[str],
    datasets_filter: Optional[List[str]],
    n_rounds: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str],
    clear_results: bool,
    test_run: bool,
    enable_validity_check: bool,
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
    total_work = sum(n * n_rounds * len(models) for n in n_per_dataset.values())
    bar = tqdm.tqdm(total=total_work, desc="Latent: iterative rounds", unit="round")

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
            state_by_query = reconstruct_state_from_records(existing_records, n_rounds)
            precompleted = sum(
                len(
                    state_by_query.get(r["query_id"], {}).get(
                        "completed_round_ids", set()
                    )
                )
                for r in rows
            )
            if precompleted:
                bar.update(precompleted)

            for row in rows:
                query_id, base_question = row["query_id"], row["base_question"]
                profession, prompt_key = row["profession"], row["prompt_key"]
                query_state = state_by_query.setdefault(
                    query_id, {"completed_round_ids": set(), "excluded_people": []}
                )
                completed, excluded_people = (
                    query_state["completed_round_ids"],
                    query_state["excluded_people"],
                )

                while True:
                    round_id = get_next_missing_round_id(completed, n_rounds)
                    if round_id is None:
                        break
                    excluded_before = list(excluded_people)
                    effective_prompt = build_iterative_prompt(
                        base_question, profession, excluded_before
                    )

                    event = {
                        "record_type": ROUND_RECORD_TYPE,
                        "timestamp": int(time.time()),
                        "run_id": run_id,
                        "model": model_name,
                        "dataset": dataset_name,
                        "query_id": query_id,
                        "prompt_key": prompt_key,
                        "profession": profession,
                        "base_question": base_question,
                        "round_id": round_id,
                        "effective_prompt": effective_prompt,
                        "excluded_people_before_round": excluded_before,
                        "response_text": "",
                        "gen_params": {},
                        "output_metadata": {},
                        "judge_output_extraction": "",
                        "person_name": "",
                        "judge_output_validity": "",
                        "is_valid": "",
                        "validity_justification": "",
                        "excluded_people_after_round": excluded_before,
                        "added_person_to_exclusion": False,
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
                        event["sample_id"] = round_id

                        person_name = "error"
                        judge_failed = False
                        try:
                            extraction_output = ask_extract_person_llm(
                                base_question, result.response_text
                            )
                            event["judge_output_extraction"] = extraction_output
                            person_name = parse_person_extraction_output(
                                extraction_output
                            ).get("person_name", "parse_error")
                        except Exception as e:
                            event["judge_output_extraction"] = f"ERROR: {e}"
                            person_name = "error"
                            judge_failed = True
                            print(
                                f"[latent] extraction judge FAILED for {model_name}/"
                                f"{dataset_name}/query_id={query_id}/round={round_id}: {e}"
                            )
                        event["person_name"] = person_name

                        if enable_validity_check and is_valid_person_name(person_name):
                            try:
                                validity_output = ask_check_person_validity_llm(
                                    person_name, profession
                                )
                                event["judge_output_validity"] = validity_output
                                parsed = parse_person_validity_output(validity_output)
                                event["is_valid"] = (
                                    "yes" if parsed.get("is_valid", False) else "no"
                                )
                                event["validity_justification"] = parsed.get(
                                    "justification", ""
                                )
                            except Exception as e:
                                event["judge_output_validity"] = f"ERROR: {e}"
                                event["is_valid"] = "error"
                                judge_failed = True
                                print(
                                    f"[latent] validity judge FAILED for {model_name}/"
                                    f"{dataset_name}/query_id={query_id}/round={round_id}: {e}"
                                )

                        should_add = False
                        if is_valid_person_name(person_name):
                            should_add = (
                                (event.get("is_valid") == "yes")
                                if enable_validity_check
                                else True
                            )
                        added = (
                            add_person_case_insensitive(excluded_people, person_name)
                            if should_add
                            else False
                        )
                        event["added_person_to_exclusion"] = added
                        event["excluded_people_after_round"] = list(excluded_people)

                        if judge_failed:
                            # Generation succeeded but a judge call raised
                            event["event_status"] = "judge_error"
                            event["error_message"] = (
                                "A judge call raised an exception this round; see "
                                "judge_output_extraction/judge_output_validity above."
                            )
                            append_jsonl_record(output_file, event)
                            bar.update(1)
                            logging.error(
                                f"Judge failed for round {round_id}, "
                                f"{model_name}/{dataset_name}/{query_id} "
                                "(left incomplete for resume)."
                            )
                            break  # leave incomplete so resume can retry

                        event["event_status"] = "ok"
                        completed.add(round_id)
                    except Exception as e:
                        event["error_message"] = str(e)
                        event["excluded_people_after_round"] = list(excluded_people)
                        append_jsonl_record(output_file, event)
                        bar.update(1)
                        logging.error(
                            f"Failed round {round_id} for {model_name}/{dataset_name}/{query_id}: {e}"
                        )
                        break  # leave incomplete so resume can retry

                    append_jsonl_record(output_file, event)
                    bar.update(1)

    max_workers = model_workers or len(models)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_model, m) for m in models]
        for _ in as_completed(futures):
            pass
    bar.close()
    logging.info("Completed latent-diversity generation.")
