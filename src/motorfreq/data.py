"""Load and prepare the French motor claims dataset (freMTPL2freq)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

RDA_URL = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2freq.rda"
CACHE_DIR = Path("data")


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    import pyreadr

    return list(pyreadr.read_r(str(path)).values())[0]


def load_raw(local_path: str | None = None, source: str = "openml") -> pd.DataFrame:
    """Load freMTPL2freq from one explicit source, so every run uses the same data.

    - "openml": the version used in the scikit-learn tutorial (678,013 policies).
    - "casdatasets": the CASdatasets R package release (677,991 policies, different claim counts).
    The two give different frequencies, so the source is fixed in the config, never mixed.
    """
    if local_path:
        return _read_file(Path(local_path)).reset_index(drop=True)
    if source == "openml":
        from sklearn.datasets import fetch_openml

        return fetch_openml(data_id=41214, as_frame=True).frame.reset_index(drop=True)
    if source == "casdatasets":
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / "freMTPL2freq.rda"
        if not path.exists():
            urllib.request.urlretrieve(RDA_URL, path)
        return _read_file(path).reset_index(drop=True)
    raise ValueError(f"unknown data source: {source!r} (use 'openml' or 'casdatasets')")


def prepare(df: pd.DataFrame, max_claims: int = 4, max_exposure: float = 1.0) -> pd.DataFrame:
    """Clean the data and add claim frequency.

    - ClaimNb is capped to limit the effect of rare extreme counts.
    - Exposure (insured time in years) is capped at one year.
    - Freq = claims per vehicle-year: the target of a frequency model.
    """
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    df["ClaimNb"] = df["ClaimNb"].astype(float).clip(upper=max_claims)
    df["Exposure"] = df["Exposure"].astype(float).clip(upper=max_exposure)
    df["Freq"] = df["ClaimNb"] / df["Exposure"]
    return df


def overall_frequency(df: pd.DataFrame) -> float:
    """Claims per vehicle-year across the whole portfolio."""
    return float(df["ClaimNb"].sum() / df["Exposure"].sum())
