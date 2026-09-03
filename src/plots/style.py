"""
Shared plotting style: model colors, labels, ordering, and rcParams.
"""

import matplotlib.pyplot as plt

MODEL_ORDER = [
    "gpt-4o-2024-08-06",
    "gpt-5.2-2025-12-11",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
]

MODEL_LABELS = {
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-5.2-2025-12-11": "GPT-5",
    "gemini-2.5-flash": "Gemini 2.5",
    "gemini-3-flash-preview": "Gemini 3",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral",
    "mistralai/Ministral-3-8B-Instruct-2512": "Ministral 3",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 2.5",
    "Qwen/Qwen3-4B": "Qwen 3",
}

# One consistent color per model
MODEL_COLORS = {
    "gpt-4o-2024-08-06": "#4C78A8",
    "gpt-5.2-2025-12-11": "#146732",
    "gemini-2.5-flash": "#72B7B2",
    "gemini-3-flash-preview": "#b95d1e",
    "meta-llama/Llama-3.1-8B-Instruct": "#ab1741",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "#9b59b6",
    "mistralai/Mistral-7B-Instruct-v0.3": "#888888",
    "mistralai/Ministral-3-8B-Instruct-2512": "#cf8b3f",
    "Qwen/Qwen2.5-3B-Instruct": "#2464a2",
    "Qwen/Qwen3-4B": "#27b5b5",
}

PROFESSION_LABELS = {
    "computer scientist": "CS",
    "woman philosopher": "Woman Phil.",
}

PROFESSIONS_ALL = [
    "chemist",
    "composer",
    "computer scientist",
    "physicist",
    "poet",
    "woman philosopher",
]
FORMATS_ALL = ["article", "bio", "name", "poem", "quote", "story"]
PROBLEM_IDS_ALL = [str(i) for i in range(1, 10)]

PROTOCOL_LABELS = {
    "accessible": "Accessible",
    "latent": "Latent",
    "multiturn": "Multi-turn",
    "multi_output": "Multi-output",
}


def profession_label(profession: str) -> str:
    return PROFESSION_LABELS.get(profession.strip().lower(), profession)


def model_label(model_version: str) -> str:
    return MODEL_LABELS.get(model_version, model_version)


def model_color(model_version: str) -> str:
    return MODEL_COLORS.get(model_version, "#333333")


def apply_paper_style() -> None:
    """Call once at the top of a notebook before plotting."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            "boxplot.flierprops.marker": "o",
            "boxplot.flierprops.markersize": 4,
        }
    )


def ordered_models(present_models) -> list:
    """Return `present_models` sorted by MODEL_ORDER, unknowns appended at the end."""
    present = set(present_models)
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered += [m for m in present if m not in MODEL_ORDER]
    return ordered
