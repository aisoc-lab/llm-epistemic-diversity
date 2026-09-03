"""
Loaders for annotated protocol outputs (used by notebooks/).
Each `load_*` function returns a normalized long-format DataFrame with a consistent column set.
"""

import logging
from typing import Callable, Dict, List, Optional, Sequence, Set

import pandas as pd

from scripts.common import PROJECT_ROOT, RESULTS_DIR

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference" / "professions"

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

PROFESSION_TO_REFERENCE_FILE = {
    "chemist": "list_of_chemists.csv",
    "composer": "list_of_composers.csv",
    "computer scientist": "list_of_computer_scientists.csv",
    "physicist": "list_of_physicists.csv",
    "poet": "list_of_poets.csv",
    "woman philosopher": "list_of_women_philosophers.csv",
}


def load_reference_names(profession: str) -> Set[str]:
    """Lowercased Wikipedia reference-list names for one profession."""
    filename = PROFESSION_TO_REFERENCE_FILE.get(profession.strip().lower())
    if filename is None:
        return set()
    path = REFERENCE_DIR / filename
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["name"].str.strip().str.lower())


def add_in_source_flag(
    df: pd.DataFrame,
    name_col: str = "resolved_name",
    profession_col: str = "profession",
) -> pd.DataFrame:
    """
    Add two derived columns, called as the last step of every professions loader:
      - `in_source`: is `name_col` present in that row's profession's reference list?
      - `format`: the prompt format ("article"/"bio"/"name"/"poem"/"quote"/"story").
    """
    df = df.copy()
    ref_cache: Dict[str, Set[str]] = {}

    def _in_source(row) -> bool:
        prof = str(row[profession_col]).strip().lower()
        if prof not in ref_cache:
            ref_cache[prof] = load_reference_names(prof)
        name = str(row[name_col]).strip().lower()
        return name in ref_cache[prof]

    df["in_source"] = df.apply(_in_source, axis=1)
    if "prompt_key" in df.columns:
        df["format"] = df["prompt_key"].astype(str).str.split("_", n=1).str[0]
    return df


def _is_valid_bool(is_valid_str) -> Optional[bool]:
    s = str(is_valid_str).strip().lower()
    if s == "yes":
        return True
    if s == "no":
        return False
    return None  # error/ parse_error/ missing


def _resolve_name_column(
    df: pd.DataFrame, fallback_col: str = "person_name", label: str = ""
) -> pd.Series:
    """
    `resolved_name`, consistent across all 4 protocols: the LLM-canonicalized name
    (`canonical_name`), so spelling variants ("Tim Berners-Lee" / "Sir Timothy Berners Lee")
    collapse to the same identity and casing matches whatever the canonicalization judge
    returned -- same convention for accessible/latent/multiturn/multi_output.

    Falls back to a lowercased `fallback_col` (with a warning) for annotated CSVs produced
    before canonicalization was applied uniformly (e.g. an old accessible-protocol file) --
    re-run `scripts/annotate.py` to pick up canonicalization for those.
    """
    if "canonical_name" in df.columns:
        return df["canonical_name"]
    msg = (
        f"No canonical_name column{' in ' + label if label else ''}; falling back to "
        f"lowercased {fallback_col} for resolved_name. Re-run scripts/annotate.py to "
        f"canonicalize and get resolved_name casing consistent with the other protocols."
    )
    print(msg)
    logging.warning(msg)
    return df[fallback_col].astype(str).str.strip().str.lower()


def _load_gathered_expanded(
    protocol: str, filename: str, models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Shared loader for the "gathered/expanded" schema common to latent, multi-turn, and
    multi-output protocols: one row per (conversation, name_rank) (already deduplicated).

    Columns expected: model, profession, prompt_key, base_question, query_id, name_rank,
    original_name, canonical_name, is_valid, validity_justification (+ optional: sample_id,
    turn_id, source_file, run_id, canonical_group_size, occurrence_count).
    """
    path = RESULTS_DIR / "annotated" / protocol / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No annotated data at {path}. Run: python scripts/annotate.py --protocol {protocol} --domain professions --models ..."
        )
    df = pd.read_csv(path)
    if models is not None:
        df = df[df["model"].isin(models)]
    df["validity_label"] = df["is_valid"].apply(_is_valid_bool)
    df["resolved_name"] = _resolve_name_column(
        df, "original_name", label=f"{protocol}/{filename}"
    )
    df = df.rename(columns={"model": "model_version", "base_question": "query"})
    if "sample_id" not in df.columns:
        df["sample_id"] = (
            0  # latent has one conversation per query; multiturn/multi_output have several
        )
    if "name_rank" not in df.columns:
        # scripts/annotate.py's multiturn path writes one row per turn with `turn_id`
        if "turn_id" in df.columns:
            df["name_rank"] = df["turn_id"]
        else:
            df["name_rank"] = df.groupby(["query_id", "sample_id"]).cumcount()
    df = add_in_source_flag(df)
    return df


def load_accessible_professions(models: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalized accessible-protocol data. `resolved_name` uses the canonicalized name.
    """
    path = RESULTS_DIR / "annotated" / "accessible" / "professions_annotated.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No annotated data at {path}. Run: python scripts/annotate.py --protocol accessible --domain professions --models ..."
        )
    df = pd.read_csv(path)
    if models is not None:
        df = df[
            df["model_version" if "model_version" in df.columns else "model"].isin(
                models
            )
        ]
    df["validity_label"] = (
        df["is_valid"].apply(_is_valid_bool)
        if df["is_valid"].dtype == object
        else df["is_valid"].astype(bool)
    )
    df["resolved_name"] = _resolve_name_column(
        df, "person_name", label="accessible professions data"
    )
    if "model_version" not in df.columns:
        df = df.rename(columns={"model": "model_version"})
    if "prompt_key" not in df.columns:
        # raw accessible-protocol records don't carry prompt_key/profession
        prompts_path = PROJECT_ROOT / "data" / "datasets" / "professions_prompts.csv"
        if prompts_path.exists():
            prompts = pd.read_csv(prompts_path)[
                ["question", "prompt_key", "profession"]
            ].rename(columns={"question": "query"})
            df = df.drop(columns=[c for c in ("profession",) if c in df.columns]).merge(
                prompts, on="query", how="left"
            )
    df = add_in_source_flag(df)
    return df


def load_accessible_professions_temperatures() -> pd.DataFrame:
    """
    Temperature ablation (T in {0.5, 1.5}; T=1.0:  `load_accessible_professions()` on gemini-3-flash-preview).
    """
    path = (
        RESULTS_DIR
        / "annotated"
        / "accessible"
        / "professions_temperatures_annotated.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"No annotated data at {path}.")
    df = pd.read_csv(path)
    df["validity_label"] = df["is_valid"].apply(_is_valid_bool)
    df["resolved_name"] = _resolve_name_column(
        df, "person_name", label="accessible temperature-ablation data"
    )
    df = df.rename(columns={"model": "model_version"})
    prompts_path = PROJECT_ROOT / "data" / "datasets" / "professions_prompts.csv"
    prompts = pd.read_csv(prompts_path)[
        ["question", "prompt_key", "profession"]
    ].rename(columns={"question": "query"})
    df = df.drop(columns=[c for c in ("profession",) if c in df.columns]).merge(
        prompts, on="query", how="left"
    )
    return add_in_source_flag(df)


def load_multiturn_professions(models: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalized multi-turn-protocol data (gathered/expanded schema)."""
    return _load_gathered_expanded("multiturn", "professions_annotated.csv", models)


def load_multi_output_professions(models: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalized multi-output-protocol data (gathered/expanded schema)."""
    return _load_gathered_expanded("multi_output", "professions_annotated.csv", models)


def load_latent_professions(
    expanded: bool = True, models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalized latent-protocol data (produced by `scripts.annotate.annotate_latent`).

    expanded=True  -> one row per (query, name_rank)
    expanded=False -> one row per (profession, canonical_name) dedup identity
    """
    if not expanded:
        path = RESULTS_DIR / "annotated" / "latent" / "professions_dedup_annotated.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No annotated data at {path}. Run: python scripts/annotate.py --protocol latent --domain professions --models ..."
            )
        df = pd.read_csv(path)
        if models is not None:
            df = df[df["model"].isin(models)] if "model" in df.columns else df
        df["validity_label"] = df["is_valid"].apply(_is_valid_bool)
        df["resolved_name"] = _resolve_name_column(
            df, "original_name", label="latent/professions_dedup_annotated.csv"
        )
        return df
    return _load_gathered_expanded(
        "latent", "professions_expanded_annotated.csv", models
    )


def load_proofs() -> pd.DataFrame:
    """Normalized proofs data (accessible protocol, judged by the proof-classification judge)."""
    path = RESULTS_DIR / "annotated" / "proofs" / "proofs_annotated.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No annotated data at {path}. Run: python scripts/annotate.py --domain proofs --models ..."
        )
    df = pd.read_csv(path, dtype={"problem_id": str}).rename(
        columns={"model": "model_version"}
    )
    # `problem_id` values are bare digits ("1".."9")
    return df


def load_wildchat() -> pd.DataFrame:
    """Normalized WildChat regime-classification data."""
    path = RESULTS_DIR / "annotated" / "wildchat" / "wildchat_questions_annotated.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No annotated data at {path}. Run: python scripts/annotate.py --domain wildchat"
        )
    return pd.read_csv(path)


def cumulative_unique_by_sample(
    df: pd.DataFrame,
    group_cols: List[str] = ("model_version", "query"),
    name_col: str = "resolved_name",
    order_col: str = "sample_id",
    filter_valid: bool = True,
) -> pd.DataFrame:
    """
    For each group (e.g. model x query), compute cumulative-unique-name count as a function of
    sample index.

    Returns a DataFrame with the original rows plus `x` (1-indexed sample position within the
    group) and `cum_unique` (cumulative unique valid names seen through that sample).
    """
    group_cols = list(group_cols)
    sub = df.copy()
    if filter_valid and "validity_label" in sub.columns:
        sub = sub[sub["validity_label"] == True]  # noqa: E712
    sub = sub[sub[name_col].notna()]
    sub = sub.sort_values(group_cols + [order_col], kind="mergesort")

    first_occurrence = ~sub.duplicated(group_cols + [name_col])
    sub["cum_unique"] = (
        first_occurrence.groupby([sub[c] for c in group_cols]).cumsum().astype(int)
    )
    sub["x"] = sub.groupby(group_cols).cumcount().add(1).astype(int)
    return sub


def try_load(
    loader: Callable[..., pd.DataFrame], *args, label: Optional[str] = None, **kwargs
) -> Optional[pd.DataFrame]:
    """
    Call a `load_*` function, catching `FileNotFoundError` (missing file) and `KeyError`/`ValueError`
    (present but unexpectedly-shaped file). Also warns if the load succeeded but came back empty.
    """
    name = label or getattr(loader, "__name__", str(loader))
    try:
        df = loader(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"Skipping {name}: {e}")
        logging.warning("Skipping %s: %s", name, e)
        return None
    except (KeyError, ValueError) as e:
        print(f"Skipping {name}: unexpected data shape ({type(e).__name__}: {e})")
        logging.warning(
            "Skipping %s: unexpected data shape (%s: %s)", name, type(e).__name__, e
        )
        return None
    if df is not None and len(df) == 0:
        print(f"{name} loaded but has 0 rows.")
        logging.warning("%s loaded but has 0 rows.", name)
    return df


def warn_incomplete_coverage(
    df: Optional[pd.DataFrame],
    label: str,
    col: str = "model_version",
    expected: Optional[Sequence[str]] = None,
) -> None:
    """
    Print a warning listing which expected values of `col` (default: models) are absent from `df`.
    """
    if df is None or len(df) == 0:
        return
    if expected is None:
        from src.plots.style import MODEL_ORDER

        expected = MODEL_ORDER
    present = set(df[col].unique())
    missing = [v for v in expected if v not in present]
    if missing:
        print(f"{label}: no data for {col} = {missing}")
