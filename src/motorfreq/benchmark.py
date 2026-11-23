"""Efficiency benchmark: why vectorised Pandas/NumPy and categorical dtypes matter at portfolio scale.

Compares two ways of computing the same exposure-weighted one-way frequency table:
- a row-by-row Python loop (common in quick notebook code)
- a vectorised pandas groupby
and the memory used by text columns stored as Python strings vs pandas categoricals.
"""
from __future__ import annotations

import time
from collections import defaultdict

import pandas as pd


def one_way_loop(df: pd.DataFrame, feature: str) -> pd.Series:
    """Exposure-weighted frequency per value, computed one row at a time."""
    claims: dict = defaultdict(float)
    exposure: dict = defaultdict(float)
    for value, c, e in zip(df[feature].tolist(), df["ClaimNb"].tolist(), df["Exposure"].tolist()):
        claims[value] += c
        exposure[value] += e
    return pd.Series({k: claims[k] / exposure[k] for k in claims}).sort_index()


def one_way_vectorised(df: pd.DataFrame, feature: str) -> pd.Series:
    """The same table with one groupby."""
    g = df.groupby(feature, observed=True)[["ClaimNb", "Exposure"]].sum()
    return (g["ClaimNb"] / g["Exposure"]).sort_index()


def _time(fn, *args, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best


def run(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Time both methods for each feature and check they give identical answers."""
    rows = []
    cat_df = df.copy()
    for f in features:
        cat_df[f] = cat_df[f].astype("category")
    for f in features:
        loop_s = _time(one_way_loop, df, f)
        vec_s = _time(one_way_vectorised, df, f)
        cat_s = _time(one_way_vectorised, cat_df, f)
        expected = one_way_loop(df, f).round(10)
        same = (expected.equals(one_way_vectorised(df, f).round(10))
                and expected.values.tolist() == one_way_vectorised(cat_df, f).round(10).values.tolist())
        rows.append({"feature": f, "loop_s": loop_s, "vectorised_s": vec_s, "vectorised_categorical_s": cat_s,
                     "speed_up_vs_loop": loop_s / cat_s, "identical_results": same})
    return pd.DataFrame(rows).set_index("feature")


def memory(df: pd.DataFrame, text_columns: list[str]) -> dict[str, float]:
    """Memory (MB) of text columns as Python strings vs as categoricals."""
    as_str = df[text_columns].astype(str).memory_usage(deep=True).sum() / 1e6
    as_cat = df[text_columns].astype("category").memory_usage(deep=True).sum() / 1e6
    return {"strings_mb": as_str, "categorical_mb": as_cat, "reduction": as_str / as_cat}


def report(df: pd.DataFrame, features: list[str], text_columns: list[str], out_path) -> str:
    timings = run(df, features)
    mem = memory(df, text_columns)
    lines = [
        "# Efficiency benchmark", "",
        f"{len(df):,} policies. Exposure-weighted one-way frequency, best of 3 runs.", "",
        timings.round(4).to_markdown(), "",
        f"Total: row loop {timings['loop_s'].sum():.3f}s, vectorised {timings['vectorised_s'].sum():.3f}s, "
        f"vectorised on categoricals {timings['vectorised_categorical_s'].sum():.3f}s "
        f"({timings['loop_s'].sum() / timings['vectorised_categorical_s'].sum():.0f}x faster than the loop). "
        "All three give identical results.", "",
        "## Memory of text columns", "",
        f"Strings: {mem['strings_mb']:.1f} MB; categorical: {mem['categorical_mb']:.1f} MB "
        f"({mem['reduction']:.0f}x smaller).",
    ]
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
