# On Epistemic Diversity in Large Language Models

This repository contains code for the paper [*On Epistemic Diversity in Large Language Models* (COLM 2026)](https://openreview.net/pdf?id=Kmrwtko9oq).

## Setup

```bash
python -m venv .epist && source .epist/bin/activate

# Requires Python 3.12
pip install -r requirements.txt

cp .env.example .env   # fill in OPENAI_API_KEY and GOOGLE_API_KEY
```

Run everything from the repository root so relative paths resolve correctly.

## 1) Datasets

1. Professions (`data/datasets/professions_prompts.csv`)
2. Math Problems (`data/datasets/math_problems.csv`)

Reference data in `data/reference/` is used to create the datasets :

- `data/reference/professions/`: 6 Wikipedia "List of X" reference lists (chemists, composers,
computer scientists, physicists, poets, women philosophers), retrieved in Jan 2026.
- `data/reference/proofs/`: 9 problem statements and reference
solution PDFs.

We also include the subset of WildChat used for annotating epistemic diversity in real user
queries (Appendix D), sampled and filtered from [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) in `data/datasets/wildchat_questions_sample.csv`.

## 2) Generation: `scripts/generate.py`

This script executes the four different interaction protocols:

- **accessible** (sampling multiple times per prompt: k=100 for professions, k=10 for proofs)
- **latent** (generate k times but exclude previously mentioned items using in-the-loop extraction and validity judging)
- **multi-turn** (generating one initial answer and 9 "Can you suggest a different person?" follow-ups)
- **multi-output** (one call, requesting 10 different responses)

Example Execution:

```bash
# Accessible
python scripts/generate.py --protocol accessible --datasets professions_prompts --models gpt-4o-2024-08-06 --n_samples 100
python scripts/generate.py --protocol accessible --datasets math_problems       --models gpt-4o-2024-08-06 --n_samples 10 --max_output_tokens 1000

# Latent
python scripts/generate.py --protocol latent --datasets professions_prompts --models gpt-4o-2024-08-06 --n_rounds 100 

# Multi-turn
python scripts/generate.py --protocol multiturn --datasets professions_prompts --models gpt-4o-2024-08-06 --n_turns 10

# Multi-output
python scripts/generate.py --protocol multi_output --datasets professions_prompts --models gpt-4o-2024-08-06 --num_responses 10
```

**Shared args** (all protocols):


| Arg                   | Default               | Meaning                                         |
| --------------------- | --------------------- | ----------------------------------------------- |
| `--protocol`          | *(required)*          | `accessible`                                    |
| `--models`            | *(required)*          | One or more model identifiers                   |
| `--datasets`          | `professions_prompts` | One or more dataset stems from `data/datasets/` |
| `--max_output_tokens` | `1000`                | default value                                   |
| `--temperature`       | `1.0`                 | default value                                   |
| `--top_p`             | `1.0`                 | default value                                   |
| `--system_prompt`     | none                  | default value                                   |
| `--clear_results`     | off                   | Delete existing output files before writing     |
| `--test_run`          | off                   | Only process the first 2 prompts per dataset    |
| `--device`            | auto                  | `cpu`                                           |
| `--model_workers`     | `len(models)`         | Parallel model workers                          |
| `--output_suffix`     | none                  | Run-id / suffix for output filenames            |


**Protocol-specific args:**

**accessible:**


| Arg               | Default | Meaning                                                 |
| ----------------- | ------- | ------------------------------------------------------- |
| `--n_samples`     | `100`   | i.i.d. samples per prompt (100 professions / 10 proofs) |
| `--hf_batch_size` | `8`     | Batch size for HuggingFace models                       |


**latent:**


| Arg                       | Default | Meaning                                          |
| ------------------------- | ------- | ------------------------------------------------ |
| `--n_rounds`              | `100`   | Exclusion rounds per prompt (default: 100)       |
| `--enable_validity_check` | off     | Run a validity judge after extraction each round |


**multiturn:**


| Arg                     | Default  | Meaning                                                      |
| ----------------------- | -------- | ------------------------------------------------------------ |
| `--n_turns`             | `10`     | Turns per conversation, incl. the first answer (default: 10) |
| `--n_samples_multiturn` | `1`      | Independent conversations per prompt                         |
| `--followup_message`    | built-in | Override the default follow-up message                       |


**multi_output:**


| Arg                          | Default  | Meaning                                                  |
| ---------------------------- | -------- | -------------------------------------------------------- |
| `--num_responses`            | `10`     | Responses requested in one call (default: 10)            |
| `--response_suffix_template` | built-in | Override the default `"{num_responses}"` suffix template |


**Used models:**

`gpt-4o-2024-08-06`, `gpt-5.2-2025-12-11`,
`gemini-2.5-flash`, `gemini-3-flash-preview`, `meta-llama/Llama-3.1-8B-Instruct`,
`meta-llama/Llama-4-Scout-17B-16E-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`,
`mistralai/Ministral-3-8B-Instruct-2512`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen3-4B`.

**Output:** `data/results/raw/<protocol>/<model>/<dataset>.jsonl`

## 3) Annotation: `scripts/annotate.py`

The interface for the four judges (person extraction, person validity, proof classification,wildchat classification).

NOTE: The math proof solution PDFs are not included in this code release. For information on how to replace them, see `data/reference/proofs/README.md`.

Example Execution:

```bash
python scripts/annotate.py --protocol accessible --domain professions --models gpt-4o-2024-08-06
python scripts/annotate.py --protocol latent      --domain professions --models gpt-4o-2024-08-06
python scripts/annotate.py --domain proofs   --models gpt-4o-2024-08-06
python scripts/annotate.py --domain wildchat
```


| Arg                | Default               | Meaning                                                                        |
| ------------------ | --------------------- | ------------------------------------------------------------------------------ |
| `--domain`         | *(required)*          | `professions`                                                                  |
| `--protocol`       | none                  | `accessible`                                                                   |
| `--models`         | `[]`                  | One or more model identifiers (required for `--domain professions` / `proofs`) whose outputs should be judged |
| `--datasets`       | `professions_prompts` | One or more dataset stems from `data/datasets/`                                |
| `--max_concurrent` | `10`                  | Max concurrent judge API calls                                                 |
| `--group_size`     | `10`                  | Names per grouped judge call (latent protocol only)                            |


**Output:** `data/results/annotated/<protocol-or-domain>/*.csv`

## 4) Figures and tables: `notebooks/`

Notebooks load annotated data via `src/data_loader.py`, compute metrics via
`src/metrics/*.py`, and plot via `src/plots/*.py`.


| Notebook                                     | Paper item(s)                        |
| -------------------------------------------- | ------------------------------------ |
| `fig02_proofs_and_entities.ipynb`            | Figure 2                             |
| `fig04_06_07_accessible_latent_grids.ipynb`  | Figures 4, 6, 7                      |
| `fig05_dominance_and_proof_categories.ipynb` | Figure 5                             |
| `fig08_proofs_grid.ipynb`                    | Figure 8                             |
| `fig09_valid_invalid_stacked.ipynb`          | Figure 9                             |
| `fig10_temperature_ablation.ipynb`           | Figure 10 (requires additional runs) |
| `fig11_protocol_comparison.ipynb`            | Figure 11                            |
| `table01_02_latent_accessible.ipynb`         | Tables 1, 2                          |
| `table03_coverage.ipynb`                     | Table 3                              |
| `table05_protocol_budget10.ipynb`            | Table 5                              |
| `wildchat_appendix_D.ipynb`                  | Appendix D                           |


Run with `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb` or open
individually.

## Contents

```
data/
  reference/          static reference material (Wikipedia lists, proof PDFs)
  datasets/            final prompt CSVs
  results/              raw + annotated outputs
scripts/
  common.py, generate.py, annotate.py
src/
  providers/    per-provider generation backends (OpenAI, Gemini, HuggingFace)
  protocols/    the 4 interaction protocols
  judges/       the 4 LLM judges
  metrics/      coverage, ratios, CI
  plots/        shared style + plotting functions
  data_loader.py
notebooks/      one notebook per figure/table group
```

