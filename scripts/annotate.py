#!/usr/bin/env python3
"""
Annotation script to run judges over raw generations for a given protocol and domain.

Four cases:

  1. --protocol {accessible, multiturn, multi_output} --domain professions
     Raw generations carry no judge output yet. Runs person_extraction, then canonicalizes
     name variants (batch grouped call), then person_validity on the canonical identity
     (per profession) on every response/turn/sub-response, writes one annotated CSV.

  2. --protocol latent --domain professions
     Raw generations already carry per-round extraction+validity from the generation loop
     itself (see src/protocols/latent.py). This step aggregates each query's final-round
     gathered names, deduplicates, canonicalizes (batch grouped call), and re-validates the
     canonical identities.

  3. --domain proofs
     Runs the proof-strategy classification judge (with the matching reference PDF attached)
     over every accessible-protocol proof generation.

  4. --domain wildchat
     Runs the WildChat regime-classification judge over the WildChat subsample directly.

Examples:
  python scripts/annotate.py --protocol accessible --domain professions --models gpt-4o-2024-08-06
  python scripts/annotate.py --protocol latent --domain professions --models gpt-4o-2024-08-06
  python scripts/annotate.py --domain proofs --models gpt-4o-2024-08-06
  python scripts/annotate.py --domain wildchat
"""

import asyncio
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import pandas as pd
from tqdm import tqdm

from scripts.common import (
    DATASETS_DIR,
    RESULTS_DIR,
    get_annotated_file,
    load_jsonl_records,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# Case 1: accessible / multiturn / multi_output -> person extraction + validity
# ---------------------------------------------------------------------------------------------


def _raw_paths(
    protocol: str, models: List[str], dataset: str
) -> List[Tuple[str, Path]]:
    out = []
    for model in models:
        safe_model = model.replace("/", "_")
        candidates = sorted(
            (RESULTS_DIR / "raw" / protocol / safe_model).glob(f"{dataset}*.jsonl")
        )
        for p in candidates:
            out.append((model, p))
    return out


def _iter_accessible_qa(path: Path):
    """(query_id placeholder via query text, question, answer, profession, prompt_key, sample_id)"""
    for rec in load_jsonl_records(path):
        yield {
            "query": rec.get("query", ""),
            "response_text": rec.get("response_text", ""),
            "sample_id": rec.get("sample_id"),
            "profession": None,
        }


def _iter_multiturn_qa(path: Path):
    for rec in load_jsonl_records(path):
        if (
            rec.get("record_type") != "multiturn_person_diversity_turn_v1"
            or rec.get("event_status") != "ok"
        ):
            continue
        yield {
            "query": rec.get("base_question", ""),
            "response_text": rec.get("response_text", ""),
            "sample_id": rec.get("sample_id"),
            "turn_id": rec.get("turn_id"),
            "profession": rec.get("profession", ""),
            "prompt_key": rec.get("prompt_key", ""),
            "query_id": rec.get("query_id"),
        }


def _iter_multi_output_qa(path: Path):
    """Each record is one completion containing several sub-responses (a numbered list, typically)."""
    for rec in load_jsonl_records(path):
        if (
            rec.get("record_type") != "multi_response_prompt_v1"
            or rec.get("event_status") != "ok"
        ):
            continue
        yield {
            "query": rec.get("base_question", ""),
            "response_text": rec.get("response_text", ""),
            "profession": rec.get("profession", ""),
            "prompt_key": rec.get("prompt_key", ""),
            "query_id": rec.get("query_id"),
        }


def _dataset_lookup(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(DATASETS_DIR / f"{dataset_name}.csv")
    return df


async def annotate_person_pipeline(
    protocol: str,
    models: List[str],
    datasets: List[str],
    max_concurrent: int,
    group_size: int,
) -> pd.DataFrame:
    from src.judges.person_extraction import (
        ask_extract_multi_person_llm_async,
        ask_extract_person_llm_async,
        parse_person_extraction_output,
    )
    from src.judges.person_validity import (
        batch_canonicalize_person_grouped_llm,
        batch_check_person_validity_grouped_llm,
        parse_person_validity_output,
    )
    from openai import AsyncOpenAI
    import os

    all_rows: List[Dict] = []
    for dataset in datasets:
        try:
            dataset_df = _dataset_lookup(dataset)
        except FileNotFoundError:
            dataset_df = None

        for model, path in _raw_paths(protocol, models, dataset):
            logger.info(f"Extracting persons: {model} / {path.name}")
            if protocol == "accessible":
                items = list(_iter_accessible_qa(path))
                for item in items:
                    if dataset_df is not None:
                        match = dataset_df[dataset_df["question"] == item["query"]]
                        if len(match):
                            item["profession"] = match.iloc[0].get("profession", "")
                            item["prompt_key"] = match.iloc[0].get("prompt_key", "")
            elif protocol == "multiturn":
                items = list(_iter_multiturn_qa(path))
            elif protocol == "multi_output":
                items = list(_iter_multi_output_qa(path))
            else:
                raise ValueError(
                    f"annotate_person_pipeline does not handle protocol={protocol}"
                )

            # --- pass 1: extraction ---
            # multi_output responses contain several sub-answers in one completion, uses the MULTI-person extraction judge.
            api_key = os.getenv("OPENAI_API_KEY")
            async with AsyncOpenAI(api_key=api_key) as client:
                semaphore = asyncio.Semaphore(max_concurrent)

                async def extract_one(item):
                    async with semaphore:
                        try:
                            raw = await ask_extract_person_llm_async(
                                item["query"],
                                item["response_text"],
                                __import__(
                                    "src.judges.person_extraction", fromlist=["PROMPT"]
                                ).PROMPT,
                                client,
                            )
                            item["person_name"] = parse_person_extraction_output(
                                raw
                            ).get("person_name", "parse_error")
                            item["judge_output_extraction"] = raw
                        except Exception as e:
                            item["person_name"] = "error"
                            item["judge_output_extraction"] = f"ERROR: {e}"
                        return item

                async def extract_multi(item):
                    async with semaphore:
                        try:
                            results = await ask_extract_multi_person_llm_async(
                                item["query"],
                                item["response_text"],
                                __import__(
                                    "src.judges.person_extraction",
                                    fromlist=["MULTI_PROMPT"],
                                ).MULTI_PROMPT,
                                client,
                            )
                        except Exception as e:
                            return [
                                dict(
                                    item,
                                    person_name="error",
                                    name_rank=0,
                                    judge_output_extraction=f"ERROR: {e}",
                                )
                            ]
                        if not results:
                            return [
                                dict(
                                    item,
                                    person_name="none",
                                    name_rank=0,
                                    judge_output_extraction=json.dumps(results),
                                )
                            ]
                        out = []
                        for rank, r in enumerate(results):
                            out.append(
                                dict(
                                    item,
                                    person_name=r.get("person_name", "parse_error"),
                                    name_rank=rank,
                                    judge_output_extraction=json.dumps(r),
                                )
                            )
                        return out

                if protocol == "multi_output":
                    expanded_lists = await asyncio.gather(
                        *(
                            extract_multi(it)
                            for it in tqdm(items, desc=f"extract {model}/{dataset}")
                        )
                    )
                    items = [row for sub in expanded_lists for row in sub]
                else:
                    items = await asyncio.gather(
                        *(
                            extract_one(it)
                            for it in tqdm(items, desc=f"extract {model}/{dataset}")
                        )
                    )

            invalid_markers = {
                "",
                "none",
                "unknown",
                "error",
                "parse_error",
                "nan",
                "n/a",
                "null",
            }

            # --- pass 1b: canonicalize name variants ---
            canon_pairs = [
                (it["person_name"], it.get("profession") or "")
                for it in items
                if it["person_name"].lower() not in invalid_markers
            ]
            if canon_pairs:
                canon_results = await batch_canonicalize_person_grouped_llm(
                    canon_pairs,
                    max_concurrent=max_concurrent,
                    group_size=group_size,
                )
                canon_by_pair = {pair: output for pair, output in canon_results}
                for it in items:
                    raw = canon_by_pair.get(
                        (it["person_name"], it.get("profession") or "")
                    )
                    if raw and not raw.startswith("ERROR:"):
                        try:
                            it["canonical_name"] = (
                                json.loads(raw).get("canonical_name")
                                or it["person_name"]
                            )
                        except Exception:
                            it["canonical_name"] = it["person_name"]
                    else:
                        it["canonical_name"] = it["person_name"]
                    it["canonical_judge_output"] = raw or ""

            # --- pass 2: grouped validity, one call per `group_size` names within a profession ---
            # Validity is checked on the canonical name
            pairs = [
                (
                    (it.get("canonical_name") or it["person_name"]),
                    it.get("profession") or "",
                )
                for it in items
                if it["person_name"].lower() not in invalid_markers
            ]
            validity_by_pair: Dict[Tuple[str, str], str] = {}
            if pairs:
                results = await batch_check_person_validity_grouped_llm(
                    pairs, max_concurrent=max_concurrent, group_size=group_size
                )
                for (name, prof), output in results:
                    validity_by_pair[(name, prof)] = output

            for item in items:
                key = (
                    item.get("canonical_name") or item["person_name"],
                    item.get("profession") or "",
                )
                validity_output = validity_by_pair.get(key)
                if validity_output is None:
                    (
                        item["is_valid"],
                        item["validity_justification"],
                        item["judge_output_validity"],
                    ) = ("", "", "")
                elif validity_output.startswith("ERROR:"):
                    (
                        item["is_valid"],
                        item["validity_justification"],
                        item["judge_output_validity"],
                    ) = ("error", "", validity_output)
                else:
                    try:
                        parsed = parse_person_validity_output(validity_output)
                        item["is_valid"] = "yes" if parsed.get("is_valid") else "no"
                        item["validity_justification"] = parsed.get("justification", "")
                    except Exception:
                        item["is_valid"] = "parse_error"
                        item["validity_justification"] = ""
                    item["judge_output_validity"] = validity_output
                item["model"] = model
                item["dataset"] = dataset
                item["protocol"] = protocol
                all_rows.append(item)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------------------------
# Case 2: latent -> aggregate final-round names, canonicalize, re-validate
# ---------------------------------------------------------------------------------------------


async def annotate_latent(
    models: List[str], datasets: List[str], max_concurrent: int, group_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (dedup_df, expanded_df)."""
    from src.judges.person_validity import (
        batch_canonicalize_person_grouped_llm,
        batch_check_person_validity_grouped_llm,
    )
    from src.protocols.latent import (
        INVALID_PERSON_NAMES,
        ROUND_RECORD_TYPE,
        is_valid_person_name,
        normalize_person_name,
    )

    final_by_key: Dict[Tuple[str, int], Dict] = {}
    for model in models:
        for dataset in datasets:
            for _model, path in _raw_paths("latent", [model], dataset):
                for rec in load_jsonl_records(path):
                    if (
                        rec.get("record_type") != ROUND_RECORD_TYPE
                        or rec.get("event_status") != "ok"
                    ):
                        continue
                    key = (str(path), rec.get("query_id"))
                    prev = final_by_key.get(key)
                    if prev is None or int(prev.get("round_id", -1)) < rec.get(
                        "round_id", -1
                    ):
                        final_by_key[key] = rec

    expanded_rows: List[Dict] = []
    for (source_file, query_id), rec in final_by_key.items():
        names = [
            n
            for n in rec.get("excluded_people_after_round", [])
            if is_valid_person_name(n)
        ]
        for name_rank, name in enumerate(names):
            expanded_rows.append(
                {
                    "source_file": source_file,
                    "query_id": query_id,
                    "name_rank": name_rank,
                    "model": rec.get("model", ""),
                    "dataset": rec.get("dataset", ""),
                    "run_id": rec.get("run_id", ""),
                    "prompt_key": rec.get("prompt_key", ""),
                    "profession": rec.get("profession", ""),
                    "base_question": rec.get("base_question", ""),
                    "original_name": normalize_person_name(name),
                }
            )

    # dedup (profession, original_name) -> occurrence_count
    dedup: Dict[Tuple[str, str], Dict] = {}
    for row in expanded_rows:
        key = (row["profession"].lower(), row["original_name"].lower())
        if key not in dedup:
            dedup[key] = {
                "profession": row["profession"],
                "original_name": row["original_name"],
                "occurrence_count": 0,
            }
        dedup[key]["occurrence_count"] += 1

    original_items = list(dedup.values())
    pairs = [(it["original_name"], it["profession"]) for it in original_items]

    # pass 1: canonicalize
    canon_results = await batch_canonicalize_person_grouped_llm(
        pairs, max_concurrent=max_concurrent, group_size=group_size
    )
    canon_by_pair = {pair: output for pair, output in canon_results}
    for it in original_items:
        raw = canon_by_pair.get((it["original_name"], it["profession"]), "")
        try:
            it["canonical_name"] = (
                json.loads(raw).get("canonical_name") or it["original_name"]
                if not raw.startswith("ERROR:")
                else it["original_name"]
            )
            it["canonical_judge_output"] = raw
        except Exception:
            it["canonical_name"], it["canonical_judge_output"] = (
                it["original_name"],
                raw,
            )

    # pass 2: validity on unique canonical identities
    canon_keys: Dict[Tuple[str, str], List[str]] = {}
    for it in original_items:
        key = (it["profession"].lower(), it["canonical_name"].lower())
        canon_keys.setdefault(key, []).append(it["original_name"].lower())
    canonical_items = [
        {"profession": p, "canonical_name": cn}
        for (p, cn) in {
            (it["profession"], it["canonical_name"]) for it in original_items
        }
    ]
    validity_pairs = [
        (it["canonical_name"], it["profession"]) for it in canonical_items
    ]
    validity_results = await batch_check_person_validity_grouped_llm(
        validity_pairs, max_concurrent=max_concurrent, group_size=group_size
    )
    validity_by_pair = {pair: output for pair, output in validity_results}

    dedup_rows = []
    for it in original_items:
        key = (it["canonical_name"], it["profession"])
        raw = validity_by_pair.get(key, "")
        is_valid, justification = "error", ""
        if raw and not raw.startswith("ERROR:"):
            try:
                parsed = json.loads(raw)
                is_valid = "yes" if parsed.get("is_valid") else "no"
                justification = parsed.get("justification", "")
            except Exception:
                is_valid = "parse_error"
        group_key = (it["profession"].lower(), it["canonical_name"].lower())
        dedup_rows.append(
            {
                "profession": it["profession"],
                "original_name": it["original_name"],
                "canonical_name": it["canonical_name"],
                "canonical_group_size": len(set(canon_keys.get(group_key, []))),
                "canonical_judge_output": it["canonical_judge_output"],
                "is_valid": is_valid,
                "validity_justification": justification,
                "validity_judge_output": raw,
                "occurrence_count": it["occurrence_count"],
            }
        )

    dedup_lookup = {
        (r["profession"].lower(), r["original_name"].lower()): r for r in dedup_rows
    }
    for row in expanded_rows:
        hit = dedup_lookup.get(
            (row["profession"].lower(), row["original_name"].lower())
        )
        if hit:
            row.update(
                {
                    k: hit[k]
                    for k in (
                        "canonical_name",
                        "canonical_group_size",
                        "canonical_judge_output",
                        "is_valid",
                        "validity_justification",
                    )
                }
            )

    return pd.DataFrame(dedup_rows), pd.DataFrame(expanded_rows)


# ---------------------------------------------------------------------------------------------
# Case 3: proofs -> proof-strategy classification
# ---------------------------------------------------------------------------------------------


async def annotate_proofs(models: List[str], max_concurrent: int) -> pd.DataFrame:
    from src.judges.proof_classification import (
        batch_proofchecker_llm,
        parse_proofchecker_output,
    )

    problems_dir = Path("data/reference/proofs/problems")
    solutions_dir = Path("data/reference/proofs/solutions")
    problem_text = {p.stem: p.read_text() for p in problems_dir.glob("*.md")}

    items: List[Tuple[str, str, str]] = []
    meta: List[Dict] = []
    for model in models:
        for _model, path in _raw_paths("accessible", [model], "math_problems"):
            for rec in load_jsonl_records(path):
                theorem = rec.get("query", "")
                problem_id = next(
                    (
                        pid
                        for pid, text in problem_text.items()
                        if text.strip() and text.strip() in theorem
                    ),
                    None,
                )
                pdf_path = (
                    str(solutions_dir / f"{problem_id}.pdf") if problem_id else None
                )
                if pdf_path is None or not Path(pdf_path).exists():
                    continue
                items.append((theorem, rec.get("response_text", ""), pdf_path))
                meta.append(
                    {
                        "model": model,
                        "problem_id": problem_id,
                        "sample_id": rec.get("sample_id"),
                    }
                )

    if not items:
        logger.warning(
            "No proof generations found under data/results/raw/accessible/*/math_problems*.jsonl"
        )
        return pd.DataFrame()

    results = await batch_proofchecker_llm(items, max_concurrent=max_concurrent)
    rows = []
    for (theorem, proof, pdf_path), m, (t2, p2, pp2, output) in zip(
        items, meta, results
    ):
        row = dict(
            m, theorem=theorem, proof=proof, pdf_path=pdf_path, judge_output=output
        )
        if output.startswith("ERROR:"):
            row["category"], row["rationale"] = "error", ""
        else:
            try:
                parsed = parse_proofchecker_output(output)
                row["category"], row["rationale"] = (
                    parsed["category"],
                    parsed["rationale"],
                )
            except Exception:
                row["category"], row["rationale"] = "parse_error", ""
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Case 4: wildchat -> regime classification
# ---------------------------------------------------------------------------------------------


async def annotate_wildchat(max_concurrent: int) -> pd.DataFrame:
    from src.judges.wildchat_classification import (
        ask_judge_llm_async,
        parse_judge_output,
    )
    from openai import AsyncOpenAI
    import os

    df = pd.read_csv(DATASETS_DIR / "wildchat_questions_sample.csv")
    api_key = os.getenv("OPENAI_API_KEY")
    async with AsyncOpenAI(api_key=api_key) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process(question):
            async with semaphore:
                try:
                    return await ask_judge_llm_async(
                        question,
                        __import__(
                            "src.judges.wildchat_classification", fromlist=["PROMPT"]
                        ).PROMPT,
                        client,
                    )
                except Exception as e:
                    return f"ERROR: {e}"

        outputs = await asyncio.gather(
            *(
                process(q)
                for q in tqdm(df["input_message"].tolist(), desc="wildchat judge")
            )
        )

    for col in [
        "judge_output",
        "number_of_responses",
        "number_of_responses_rationale",
        "specification",
        "specification_rationale",
    ]:
        df[col] = ""
    for i, output in enumerate(outputs):
        df.at[i, "judge_output"] = output
        if not output.startswith("ERROR:"):
            try:
                parsed = parse_judge_output(output)
                for k in [
                    "number_of_responses",
                    "number_of_responses_rationale",
                    "specification",
                    "specification_rationale",
                ]:
                    df.at[i, k] = parsed[k]
            except Exception:
                pass
    return df


# ---------------------------------------------------------------------------------------------
# Merge-and-write: accumulate across separate invocations instead of overwriting
# ---------------------------------------------------------------------------------------------


def merge_and_write_csv(
    new_df: pd.DataFrame, out_path, key_cols: List[str]
) -> pd.DataFrame:
    """
    Write `new_df` to `out_path`, merging with whatever's already there instead of overwriting it.

    `key_cols` must be columns present in `new_df`. Missing ones are dropped with a warning.
    """
    key_cols = [c for c in key_cols if c in new_df.columns]
    if not key_cols:
        logger.warning(
            f"No valid key columns for {out_path}; falling back to overwrite (no merge)."
        )
        new_df.to_csv(out_path, index=False)
        return new_df

    if Path(out_path).exists():
        existing_df = pd.read_csv(out_path, dtype=str)
        combined = pd.concat([existing_df, new_df.astype(str)], ignore_index=True)
        before = len(existing_df)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        n_new = len(combined) - before
        n_updated = len(new_df) - max(n_new, 0)
        logger.info(
            f"Merged into {out_path}: {max(n_new, 0)} new row(s), ~{max(n_updated, 0)} updated existing row(s)."
        )
    else:
        combined = new_df

    combined.to_csv(out_path, index=False)
    return combined


def main() -> None:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domain", required=True, choices=["professions", "proofs", "wildchat"]
    )
    parser.add_argument(
        "--protocol",
        choices=["accessible", "latent", "multiturn", "multi_output"],
        help="Required for --domain professions.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Required for --domain professions/proofs.",
    )
    parser.add_argument("--datasets", nargs="+", default=["professions_prompts"])
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument(
        "--group_size",
        type=int,
        default=10,
        help="Names per grouped judge call (latent protocol only).",
    )
    args = parser.parse_args()

    if args.domain == "professions":
        if not args.protocol or not args.models:
            parser.error(
                "--protocol and --models are required for --domain professions"
            )
        if args.protocol == "latent":
            dedup_df, expanded_df = asyncio.run(
                annotate_latent(
                    args.models, args.datasets, args.max_concurrent, args.group_size
                )
            )
            dedup_path = get_annotated_file("latent", "professions_dedup")
            expanded_path = get_annotated_file("latent", "professions_expanded")
            merge_and_write_csv(
                dedup_df, dedup_path, key_cols=["profession", "original_name"]
            )
            merge_and_write_csv(
                expanded_df, expanded_path, key_cols=["model", "query_id", "name_rank"]
            )
            logger.info(
                f"Merged {len(dedup_df)} new dedup row(s) -> {dedup_path}, "
                f"{len(expanded_df)} new expanded row(s) -> {expanded_path}"
            )
        else:
            df = asyncio.run(
                annotate_person_pipeline(
                    args.protocol,
                    args.models,
                    args.datasets,
                    args.max_concurrent,
                    args.group_size,
                )
            )
            out_path = get_annotated_file(args.protocol, "professions")
            key_cols = {
                "accessible": ["model", "query", "sample_id"],
                "multiturn": ["model", "query_id", "sample_id", "turn_id"],
                "multi_output": ["model", "query_id", "name_rank"],
            }[args.protocol]
            merge_and_write_csv(df, out_path, key_cols=key_cols)
            logger.info(f"Merged {len(df)} new row(s) -> {out_path}")
    elif args.domain == "proofs":
        if not args.models:
            parser.error("--models is required for --domain proofs")
        df = asyncio.run(annotate_proofs(args.models, args.max_concurrent))
        out_path = get_annotated_file("proofs", "proofs")
        merge_and_write_csv(df, out_path, key_cols=["model", "problem_id", "sample_id"])
        logger.info(f"Merged {len(df)} new row(s) -> {out_path}")
    elif args.domain == "wildchat":
        df = asyncio.run(annotate_wildchat(args.max_concurrent))
        out_path = get_annotated_file("wildchat", "wildchat_questions")
        merge_and_write_csv(df, out_path, key_cols=["conversation_hash", "turn"])
        logger.info(f"Merged {len(df)} new row(s) -> {out_path}")


if __name__ == "__main__":
    main()
