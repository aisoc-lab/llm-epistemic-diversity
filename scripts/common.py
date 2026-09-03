"""
Shared I/O helpers and result dataclasses
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Fixed project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = Path(
    os.getenv("EPISTEMIC_DATASETS_DIR", PROJECT_ROOT / "data" / "datasets")
)
RESULTS_DIR = Path(
    os.getenv("EPISTEMIC_RESULTS_DIR", PROJECT_ROOT / "data" / "results")
)
PROMPT_COLUMN = (
    "question"  # column name in dataset CSVs that holds the prompt/question text
)


@dataclass
class GenerationParameters:
    """Parameters used for a single generation call."""

    max_output_tokens: int
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    system_prompt: Optional[str] = None


@dataclass
class OutputMetadata:
    """Metadata attached to a single generation call."""

    input_tokens: int
    output_tokens: int
    finish_reason: str
    timestamp: int
    response_time_ms: int


@dataclass
class LLMResult:
    """Unified result object returned by every provider's `query_model`."""

    query: str
    model_version: str
    response_text: str
    gen_params: GenerationParameters
    output_metadata: OutputMetadata
    sample_id: int = 0


def get_output_file(
    model_id: str,
    dataset_name: str,
    protocol: str,
    create_dir: bool = False,
    delete_existing_file: bool = False,
    return_line_count: bool = False,
    output_suffix: Optional[str] = None,
) -> Path | tuple[Path, int]:
    """
    Returns the raw-generation output file path: data/results/raw/<protocol>/<model>/<dataset>[_suffix].jsonl
    """
    safe_model_id = model_id.replace("/", "_")
    output_dir = RESULTS_DIR / "raw" / protocol / safe_model_id

    if create_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        output_dir / f"{dataset_name}_{output_suffix}.jsonl"
        if output_suffix
        else output_dir / f"{dataset_name}.jsonl"
    )

    if return_line_count:
        line_count = 0
        if filename.exists():
            with open(filename, "r") as f:
                line_count = sum(1 for _ in f)

    if delete_existing_file and filename.exists():
        logging.warning(
            f"Deleting existing file {filename} before writing new results."
        )
        filename.unlink()

    return filename if not return_line_count else (filename, line_count)


def get_annotated_file(protocol: str, domain: str) -> Path:
    """Returns the canonical annotated-output path: data/results/annotated/<protocol>/<domain>_annotated.csv"""
    output_dir = RESULTS_DIR / "annotated" / protocol
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{domain}_annotated.csv"


def get_datasets(datasets_dir: Optional[Path] = None) -> List[Dict[str, pd.DataFrame]]:
    """
    Returns one dict per CSV found in `datasets_dir` (default: data/datasets/)
    """
    datasets_path = Path(datasets_dir) if datasets_dir is not None else DATASETS_DIR
    if not datasets_path.exists():
        logging.warning(f"Datasets directory {datasets_path} does not exist.")
        return []

    datasets = []
    for path in sorted(datasets_path.glob("*.csv")):
        try:
            df = pd.read_csv(path)
            if PROMPT_COLUMN not in df.columns:
                if "prompt" in df.columns:
                    df = df.rename(columns={"prompt": PROMPT_COLUMN})
                else:
                    continue  # not a prompt dataset
            datasets.append({path.stem: df})
        except Exception as e:
            logging.warning(f"Failed to load dataset {path}: {e}. Skipping.")

    return datasets


_append_locks: Dict[str, threading.Lock] = {}
_append_locks_guard = threading.Lock()


def append_jsonl_record(file_path: Path, record: Dict[str, Any] | LLMResult) -> None:
    """Append one JSON record to `file_path`"""
    if isinstance(record, LLMResult):
        record = asdict(record)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    path_key = str(file_path)
    with _append_locks_guard:
        if path_key not in _append_locks:
            _append_locks[path_key] = threading.Lock()
    with _append_locks[path_key]:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_records(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from `file_path`, skipping malformed lines"""
    if not file_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                logging.warning(
                    f"Skipping malformed JSONL line {line_idx + 1} in {file_path}"
                )
    return records
