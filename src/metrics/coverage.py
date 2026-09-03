"""
Compute coverage of Wikipedia reference lists by profession, aggregated across all models and prompt formats.
"""

import pandas as pd


def coverage_by_profession(accessible_df: pd.DataFrame) -> pd.DataFrame:
    """
    `accessible_df` must have columns: profession, in_source (bool), validity_label (bool).

    Returns one row per profession with:
      pct_found_in_source          -- named person is on the Wikipedia reference list
      pct_valid_not_in_source      -- named person is judged valid/well-known but absent from list
      pct_invalid_not_in_source    -- named person is judged invalid (or unresolved) and absent
      n_unique_named                -- unique names encountered (any validity)
      n_unique_in_source             -- unique reference-list names actually named
    """
    rows = []
    for profession, sub in accessible_df.groupby("profession"):
        n = len(sub)
        if n == 0:
            continue
        rows.append(
            {
                "profession": profession,
                "pct_found_in_source": 100.0 * (sub["in_source"] == True).sum() / n,
                "pct_valid_not_in_source": 100.0
                * ((~sub["in_source"]) & (sub["validity_label"] == True)).sum()
                / n,
                "pct_invalid_not_in_source": 100.0
                * ((~sub["in_source"]) & (sub["validity_label"] != True)).sum()
                / n,
                "n_unique_named": sub["resolved_name"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values("profession").reset_index(drop=True)


def unique_reference_coverage_pct(
    accessible_df: pd.DataFrame, reference_sizes: dict
) -> pd.DataFrame:
    """
    Out of the distinct reference-list names, how many were ever named by any model/format?
    `n_unique_in_source / |reference list|`

    `reference_sizes` maps profession -> reference list size
    """
    rows = []
    for profession, sub in accessible_df.groupby("profession"):
        in_source_names = set(
            sub.loc[sub["in_source"] == True, "resolved_name"].str.strip().str.lower()
        )
        ref_size = reference_sizes.get(profession.strip().lower(), 0)
        rows.append(
            {
                "profession": profession,
                "unique_named_in_source": len(in_source_names),
                "reference_list_size": ref_size,
                "coverage_pct": (
                    100.0 * len(in_source_names) / ref_size
                    if ref_size
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("profession").reset_index(drop=True)
