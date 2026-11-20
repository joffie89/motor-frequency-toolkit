import numpy as np
import pytest

from motorfreq import data, evaluate, features


def test_time_split_has_no_overlap(synthetic_raw):
    df = features.add_periods(data.prepare(synthetic_raw), 4, seed=0)
    train, test = evaluate.time_split(df, 4)
    assert train["Period"].max() < 4
    assert set(test["Period"]) == {4}


def test_drift_only_changes_target_segment(synthetic_raw):
    df = features.add_periods(data.prepare(synthetic_raw), 4, seed=0)
    drifted = features.inject_drift(df, "VehBrand", "B12", 4, 1.0, seed=0)
    untouched = ~((df["VehBrand"] == "B12") & (df["Period"] == 4))
    assert (drifted.loc[untouched, "ClaimNb"] == df.loc[untouched, "ClaimNb"]).all()
    assert drifted["ClaimNb"].sum() > df["ClaimNb"].sum()


def test_area_groups_cover_every_level(synthetic_raw):
    df = data.prepare(synthetic_raw)
    mapping = features.fit_area_groups(df, "Region", n_groups=3, credibility_k=50)
    assert set(mapping) == set(df["Region"].unique())
    assert set(mapping.values()) == {0, 1, 2}
    grouped = features.apply_area_groups(df, "Region", mapping)
    assert grouped["AreaGroup"].notna().all()


def test_perfect_predictions_are_calibrated():
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1, 200_000)
    rate = rng.choice([0.05, 0.2], len(w))
    claims = rng.poisson(rate * w)
    assert evaluate.observed_expected(claims, rate, w) == pytest.approx(1.0, abs=0.02)
    assert evaluate.lift(claims, rate, w) > 2
