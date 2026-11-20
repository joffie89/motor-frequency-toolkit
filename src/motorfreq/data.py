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


def load_raw(local_path: str | None = None) -> pd.DataFrame:
    """Load freMTPL2freq.

    Order: a local file if given; otherwise OpenML; otherwise the CASdatasets copy on GitHub.
    """
    if local_path:
        return _read_file(Path(local_path)).reset_index(drop=True)
    try:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=41214, as_frame=True, n_retries=1).frame
    except Exception:
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / "freMTPL2freq.rda"
        if not path.exists():
            urllib.request.urlretrieve(RDA_URL, path)
        df = _read_file(path)
    return df.reset_index(drop=True)


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
