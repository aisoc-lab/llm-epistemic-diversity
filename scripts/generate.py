#!/usr/bin/env python3
"""
Generation script to run any of the 4 interaction protocols against any model/dataset.

Examples:
  # Accessible: k=100 samples/prompt, professions, one model
  python scripts/generate.py --protocol accessible --datasets professions_prompts \\
      --models gpt-4o-2024-08-06 --n_samples 100

  # Accessible: k=10 samples/prompt, proofs
  python scripts/generate.py --protocol accessible --datasets math_problems \\
      --models gpt-4o-2024-08-06 --n_samples 10 --max_output_tokens 1000

  # Latent: 100 exclusion rounds, professions
  python scripts/generate.py --protocol latent --datasets professions_prompts \\
      --models gpt-4o-2024-08-06 --n_rounds 100

  # Multi-turn: 10 turns (1 + 9 follow-ups)
  python scripts/generate.py --protocol multiturn --datasets professions_prompts \\
      --models gpt-4o-2024-08-06 --n_turns 10

  # Multi-output: 1 call asking for 10 responses
  python scripts/generate.py --protocol multi_output --datasets professions_prompts \\
      --models gpt-4o-2024-08-06 --num_responses 10
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.basicConfig(level=logging.INFO)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--protocol",
        required=True,
        choices=["accessible", "latent", "multiturn", "multi_output"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model identifiers (e.g. gpt-4o-2024-08-06, gemini-2.5-flash, meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["professions_prompts"],
        help="Dataset stems from data/datasets/ (e.g. professions_prompts, math_problems).",
    )
    parser.add_argument("--max_output_tokens", type=int, default=1000)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="default: 1.0 for all protocols/models.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="default: 1.0 for all protocols/models.",
    )
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument(
        "--clear_results",
        action="store_true",
        help="Delete existing output file(s) before writing.",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only process the first 2 prompts per dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for HuggingFace models.",
    )
    parser.add_argument(
        "--model_workers",
        type=int,
        default=None,
        help="Parallel model workers. Defaults to len(models).",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Run-id / suffix for output filenames.",
    )

    # accessible-only
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="[accessible] i.i.d. samples per prompt. default: 100 (professions) / 10 (proofs).",
    )
    parser.add_argument(
        "--hf_batch_size",
        type=int,
        default=8,
        help="[accessible] Batch size for HuggingFace models.",
    )

    # latent-only
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=100,
        help="[latent] Exclusion rounds per prompt. default: 100.",
    )
    parser.add_argument(
        "--enable_validity_check",
        action="store_true",
        help="[latent] Run a validity judge after extraction each round (default: False).",
    )

    # multiturn-only
    parser.add_argument(
        "--n_turns",
        type=int,
        default=10,
        help="[multiturn] Turns per conversation, incl. the first answer. default: 10.",
    )
    parser.add_argument(
        "--n_samples_multiturn",
        type=int,
        default=1,
        dest="n_samples_multiturn",
        help="[multiturn] Independent conversations per prompt. default: 1.",
    )
    parser.add_argument(
        "--followup_message",
        type=str,
        default=None,
        help="[multiturn] Override the default follow-up message.",
    )

    # multi_output-only
    parser.add_argument(
        "--num_responses",
        type=int,
        default=10,
        help="[multi_output] Responses requested in one call. default: 10.",
    )
    parser.add_argument(
        "--response_suffix_template",
        type=str,
        default=None,
        help="[multi_output] Override the default '{num_responses}' suffix template.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.protocol == "accessible":
        from src.protocols.accessible import run_accessible

        run_accessible(
            models=args.models,
            datasets_filter=args.datasets,
            n_samples=args.n_samples,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
            clear_results=args.clear_results,
            test_run=args.test_run,
            device=args.device,
            model_workers=args.model_workers,
            hf_batch_size=args.hf_batch_size,
            output_suffix=args.output_suffix,
        )
    elif args.protocol == "latent":
        from src.protocols.latent import run_latent

        run_latent(
            models=args.models,
            datasets_filter=args.datasets,
            n_rounds=args.n_rounds,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
            clear_results=args.clear_results,
            test_run=args.test_run,
            enable_validity_check=args.enable_validity_check,
            model_workers=args.model_workers,
            device=args.device,
            output_suffix=args.output_suffix,
        )
    elif args.protocol == "multiturn":
        from src.protocols.multiturn import DEFAULT_FOLLOWUP_MESSAGE, run_multiturn

        run_multiturn(
            models=args.models,
            datasets_filter=args.datasets,
            n_turns=args.n_turns,
            n_samples=args.n_samples_multiturn,
            followup_message=args.followup_message or DEFAULT_FOLLOWUP_MESSAGE,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
            clear_results=args.clear_results,
            test_run=args.test_run,
            model_workers=args.model_workers,
            device=args.device,
            output_suffix=args.output_suffix,
        )
    elif args.protocol == "multi_output":
        from src.protocols.multi_output import (
            DEFAULT_RESPONSE_SUFFIX_TEMPLATE,
            run_multi_output,
        )

        run_multi_output(
            models=args.models,
            datasets_filter=args.datasets,
            num_responses=args.num_responses,
            response_suffix_template=args.response_suffix_template
            or DEFAULT_RESPONSE_SUFFIX_TEMPLATE,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
            clear_results=args.clear_results,
            test_run=args.test_run,
            model_workers=args.model_workers,
            device=args.device,
            output_suffix=args.output_suffix,
        )
    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")


if __name__ == "__main__":
    main()
