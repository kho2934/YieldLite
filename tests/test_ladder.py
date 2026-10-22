"""Tests for yieldlite.ladder."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from yieldlite.features import VarietyPooler
from yieldlite.ladder import RUNGS, PoolVarieties, build_rung
from yieldlite.models import build_m1

_SEASONS = ["DX", "HT", "TD"]
_DISTRICTS = ["CD", "OM", "TN"]


def _survey_rows(n_per_cell: int = 8, seed: int = 0) -> pd.DataFrame:
    """Create survey-like rows with raw variety columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, season in enumerate(_SEASONS):
        for d, district in enumerate(_DISTRICTS):
            for i in range(n_per_cell):
                rows.append(
                    {
                        "season": season,
                        "district_code": district,
                        "variety_raw": ["A", "B", "C"][i % 3],
                        "variety_flag": "known",
                        "npk_total_kg_ha": 100.0 + 15.0 * i + 5.0 * d,
                        "yield_t_ha": 5.0 + 0.5 * s + 0.2 * d + rng.normal(0, 0.1),
                    }
                )
    return pd.DataFrame(rows)


def _small_variety_frame(varieties, flags=None) -> pd.DataFrame:
    flags = flags or ["known"] * len(varieties)
    return pd.DataFrame({"variety_raw": varieties, "variety_flag": flags})


def test_rare_varieties_are_pooled_using_the_fitted_rows_only():
    train = _small_variety_frame(["A", "A", "A", "B"])
    test = _small_variety_frame(["A", "B", "C"])

    step = PoolVarieties(min_count=3).fit(train)
    pooled = step.transform(test)["variety_pooled"]

    # A has three training rows, B has one, and C was never seen.
    assert pooled.tolist() == ["A", "pooled", "pooled"]


def test_unclear_variety_labels_are_pooled_even_when_common():
    train = _small_variety_frame(
        ["A", "A", "A"],
        ["known", "known", "known"],
    )
    test = _small_variety_frame(["A", "A"], ["known", "unclear"])

    pooled = PoolVarieties(min_count=3).fit(train).transform(test)

    assert pooled["variety_pooled"].tolist() == ["A", "pooled"]


def test_pooling_matches_the_variety_pooler_it_wraps():
    frame = _survey_rows()
    pooler = VarietyPooler(5).fit(frame["variety_raw"], frame["variety_flag"])

    pooled = PoolVarieties(min_count=5).fit(frame).transform(frame)

    pd.testing.assert_series_equal(
        pooled["variety_pooled"],
        pooler.transform(frame["variety_raw"], frame["variety_flag"]),
        check_names=False,
    )


def test_transform_keeps_the_original_columns_and_does_not_edit_the_input():
    frame = _survey_rows()
    before = frame.copy()

    out = PoolVarieties(min_count=5).fit(frame).transform(frame)

    assert set(frame.columns) | {"variety_pooled"} == set(out.columns)
    pd.testing.assert_frame_equal(frame, before)


def test_transform_before_fit_fails():
    with pytest.raises(NotFittedError):
        PoolVarieties().transform(_small_variety_frame(["A"]))


def test_missing_variety_columns_are_rejected():
    with pytest.raises(ValueError, match="variety_flag"):
        PoolVarieties().fit(pd.DataFrame({"variety_raw": ["A"]}))


def test_input_must_be_a_dataframe():
    with pytest.raises(TypeError):
        PoolVarieties().fit(np.zeros((3, 2)))


def test_clone_keeps_the_setting_and_drops_the_fit():
    fitted = PoolVarieties(min_count=7).fit(_small_variety_frame(["A"] * 7))

    copy = clone(fitted)

    assert copy.min_count == 7
    assert not hasattr(copy, "pooler_")


def test_rungs_are_listed_in_ladder_order():
    assert RUNGS == ("m0", "m1")


def test_unknown_rung_is_rejected_with_the_choices():
    with pytest.raises(ValueError, match="m0"):
        build_rung("m9")


@pytest.mark.parametrize("name", ["m0", "m1"])
def test_every_rung_fits_and_predicts_from_raw_columns(name):
    frame = _survey_rows()

    model = build_rung(name, min_variety_count=5)
    model.fit(frame, frame["yield_t_ha"])
    predicted = np.asarray(model.predict(frame), dtype=float)

    assert predicted.shape == (len(frame),)
    assert np.isfinite(predicted).all()
    assert np.abs(predicted - frame["yield_t_ha"]).mean() < 0.5


def test_m1_pipeline_matches_pooling_by_hand():
    train = _survey_rows(seed=1)
    test = _survey_rows(seed=2)

    pooler = VarietyPooler(10).fit(train["variety_raw"], train["variety_flag"])
    by_hand = build_m1()
    by_hand.fit(
        train.assign(
            variety_pooled=pooler.transform(
                train["variety_raw"],
                train["variety_flag"],
            )
        ),
        train["yield_t_ha"],
    )
    expected = by_hand.predict(
        test.assign(
            variety_pooled=pooler.transform(
                test["variety_raw"],
                test["variety_flag"],
            )
        )
    )

    pipeline = build_rung("m1", min_variety_count=10)
    pipeline.fit(train, train["yield_t_ha"])

    np.testing.assert_allclose(pipeline.predict(test), expected)


def test_each_call_builds_a_new_unfitted_pipeline():
    first = build_rung("m1")
    second = build_rung("m1")

    first.fit(_survey_rows(), _survey_rows()["yield_t_ha"])

    assert first is not second
    assert not hasattr(second.named_steps["pool"], "pooler_")
