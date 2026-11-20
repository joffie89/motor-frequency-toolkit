import pandas as pd
import pytest

from motorfreq import data, validate


def test_prepare_caps_and_adds_frequency():
    raw = pd.DataFrame({"ClaimNb": [0, 2, 9], "Exposure": [0.5, 1.4, 1.0]})
    df = data.prepare(raw, max_claims=4, max_exposure=1.0)
    assert df["ClaimNb"].max() == 4
    assert df["Exposure"].max() == 1.0
    assert df["Freq"].tolist() == [0.0, 2.0, 4.0]


def test_overall_frequency_is_claims_per_vehicle_year():
    df = pd.DataFrame({"ClaimNb": [1, 0, 1], "Exposure": [0.5, 0.5, 1.0]})
    assert data.overall_frequency(df) == pytest.approx(1.0)


def test_validation_catches_bad_exposure():
    bad = pd.DataFrame({"ClaimNb": [0, 1], "Exposure": [0.0, 1.5]})
    with pytest.raises(validate.DataQualityError):
        validate.validate(bad)


def test_validation_passes_clean_data(synthetic_raw):
    df = data.prepare(synthetic_raw)
    assert validate.check(df, ["VehBrand"]) == []
