"""
Accessible diversity protocol (k independent i.i.d. samples per prompt)
k=100 for professions, k=10 for proofs
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import tqdm
import json

from scripts.common import (
    PROMPT_COLUMN,
    append_jsonl_record,
    get_datasets,
    get_output_file,
)
from src.providers import (
    get_temperature_for_model,
    get_top_p_for_model,
    is_huggingface_model,
    query_huggingface_batch,
    query_model_with_params,
)

PROTOCOL_NAME = "accessible"
TEST_RUN_NUM_QUERIES = 2


def load_existing_sample_ids(output_file, queries: List[str]) -> Dict[str, set]:
    """Preserve gaps, fill in missing sample IDs."""
    if not output_file.exists():
        return {q: set() for q in queries}
    remaining = set(queries)
    existing: Dict[str, set] = {q: set() for q in remaining}
    with open(output_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = data.get("query")
            if q not in remaining or "sample_id" not in data:
                continue
            try:
                existing[q].add(int(data["sample_id"]))
            except Exception:
                continue
    return existing


def get_next_missing_sample_id(existing_ids: set, n_samples: int) -> Optional[int]:
    for sid in range(n_samples):
        if sid not in existing_ids:
            return sid
    return None


def run_accessible(
    models: List[str],
    datasets_filter: Optional[List[str]],
    n_samples: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str],
    clear_results: bool,
    test_run: bool,
    device: Optional[str],
    model_workers: Optional[int],
    hf_batch_size: int,
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

    n_per_dataset = {}
    for dataset in datasets:
        name, df = list(dataset.items())[0]
        n = len(df[PROMPT_COLUMN])
        n_per_dataset[name] = min(TEST_RUN_NUM_QUERIES, n) if test_run else n
    total_work = sum(n * n_samples * len(models) for n in n_per_dataset.values())
    bar = tqdm.tqdm(
        total=total_work, desc="Accessible: generating samples", unit="sample"
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
                output_suffix=output_suffix,
            )
            queries = df[PROMPT_COLUMN].tolist()
            if test_run:
                queries = queries[:TEST_RUN_NUM_QUERIES]
            existing = load_existing_sample_ids(output_file, queries)

            if is_huggingface_model(model_name):
                _run_hf_batched(
                    model_name,
                    queries,
                    existing,
                    output_file,
                    n_samples,
                    max_output_tokens,
                    temperature,
                    top_p,
                    system_prompt,
                    hf_batch_size,
                    bar,
                    llama_kwargs,
                )
                continue

            for prompt in queries:
                existing_ids = existing.get(prompt, set())
                missing = [sid for sid in range(n_samples) if sid not in existing_ids]
                if not missing:
                    bar.update(n_samples)
                    continue
                for sample_id in missing:
                    try:
                        model_temp = get_temperature_for_model(model_name, temperature)
                        model_top_p = get_top_p_for_model(model_name, top_p)
                        result, _ = query_model_with_params(
                            prompt=prompt,
                            model_name=model_name,
                            max_output_tokens=max_output_tokens,
                            temperature=model_temp,
                            top_p=model_top_p,
                            system_prompt=system_prompt,
                            **llama_kwargs,
                        )
                        result.sample_id = sample_id
                        append_jsonl_record(output_file, result)
                    except Exception as e:
                        logging.error(
                            f"Error sample {sample_id} for '{prompt[:50]}...' ({model_name}): {e}"
                        )
                    bar.update(1)

    max_workers = model_workers or len(models)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_model, m) for m in models]
        for _ in as_completed(futures):
            pass
    bar.close()
    logging.info("Completed accessible-diversity generation.")


def _run_hf_batched(
    model_name,
    queries,
    existing,
    output_file,
    n_samples,
    max_output_tokens,
    temperature,
    top_p,
    system_prompt,
    hf_batch_size,
    bar,
    llama_kwargs,
) -> None:
    """Batched HF generation: one sample per prompt per batch until each prompt reaches n_samples."""
    existing_per_query = [existing.get(q, set()) for q in queries]
    next_sample_id = [
        get_next_missing_sample_id(ids, n_samples) for ids in existing_per_query
    ]
    if all(sid is None for sid in next_sample_id):
        bar.update(len(queries) * n_samples)
        return

    while True:
        need_idxs = [i for i, sid in enumerate(next_sample_id) if sid is not None]
        if not need_idxs:
            break
        batch_idxs = need_idxs[: max(1, hf_batch_size)]
        batch_prompts = [queries[i] for i in batch_idxs]
        try:
            model_temp = get_temperature_for_model(model_name, temperature)
            model_top_p = get_top_p_for_model(model_name, top_p)
            batch_results = query_huggingface_batch(
                prompts=batch_prompts,
                model_name=model_name,
                max_output_tokens=max_output_tokens,
                temperature=model_temp,
                top_p=model_top_p,
                system_prompt=system_prompt,
                device=llama_kwargs.get("device"),
            )
            for local_idx, (result, _) in enumerate(batch_results):
                global_idx = batch_idxs[local_idx]
                sample_id = next_sample_id[global_idx]
                if sample_id is None:
                    continue
                result.sample_id = sample_id
                append_jsonl_record(output_file, result)
                existing_per_query[global_idx].add(sample_id)
                next_sample_id[global_idx] = get_next_missing_sample_id(
                    existing_per_query[global_idx], n_samples
                )
            bar.update(len(batch_results))
        except Exception as e:
            logging.error(
                f"Error in HF batch for {model_name}: {e}. Falling back to per-prompt."
            )
            for i in batch_idxs:
                if next_sample_id[i] is None:
                    continue
                try:
                    result, _ = query_model_with_params(
                        prompt=queries[i],
                        model_name=model_name,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        system_prompt=system_prompt,
                        **llama_kwargs,
                    )
                    sample_id = next_sample_id[i]
                    result.sample_id = sample_id
                    append_jsonl_record(output_file, result)
                    existing_per_query[i].add(sample_id)
                    next_sample_id[i] = get_next_missing_sample_id(
                        existing_per_query[i], n_samples
                    )
                except Exception as e2:
                    logging.error(
                        f"HF single-inference fallback failed for {model_name}: {e2}"
                    )
                bar.update(1)
