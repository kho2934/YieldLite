"""Tests for deriving the coverage threshold P0 from interval-study results."""

import json

import numpy as np
import pandas as pd
import pytest

from yieldlite.thresholds import coverage_check, derive_thresholds


def _setting(model, level, p, h, coverage, se):
    """Create one row of an interval summary; no coverage means unsupported."""
    supported = coverage is not None
    return {
        "model": model,
        "level": level,
        "p": p,
        "h": h,
        "supported": supported,
        "coverage": coverage if supported else np.nan,
        "coverage_se": se if supported else np.nan,
    }


def _summary(coverage_by_p, se=0.01, level=0.8):
    """Create a summary with the same coverage for every model and cap."""
    return pd.DataFrame(
        [
            _setting(model, level, p, h, coverage, se)
            for p, coverage in coverage_by_p.items()
            for model in ("m0", "m1")
            for h in (20.0, np.inf)
        ]
    )


def _degradation_results():
    """Create degradation results in which M1 beats M0 everywhere."""
    rows = []
    for panel, offset in zip("ABCDEFGH", np.linspace(-0.01, 0.01, 8), strict=True):
        for p in (3, 4):
            for h in (20.0, np.inf):
                rows.append((panel, p, h, "m0", 1.0))
                rows.append((panel, p, h, "m1", 0.9 - offset))
    return pd.DataFrame(rows, columns=["held_out", "p", "h", "model", "mae"])


def _p0(coverage_by_p, **kwargs):
    output = derive_thresholds(
        _degradation_results(),
        intervals=_summary(coverage_by_p),
        **kwargs,
    )
    return output["thresholds"]["P0"]


def test_panel_count_passes_when_every_setting_clears_the_level():
    check = coverage_check(_summary({4: 0.95, 5: 0.9}))

    assert check["p"].tolist() == [4, 5]
    assert check["n_settings"].tolist() == [4, 4]
    assert check["n_passing"].tolist() == [4, 4]
    assert check["passes"].tolist() == [True, True]
    np.testing.assert_allclose(check["lowest_coverage"], [0.95, 0.9])


def test_unsupported_level_fails_the_panel_count():
    check = coverage_check(_summary({3: None, 4: 0.95}))

    assert check["passes"].tolist() == [False, True]
    assert check["n_passing"].tolist() == [0, 4]
    assert np.isnan(check["lowest_coverage"].iloc[0])


def test_one_weak_setting_fails_the_whole_panel_count():
    summary = _summary({5: 0.95})
    weak = (summary["model"] == "m1") & (summary["h"] == 20.0)
    summary.loc[weak, "coverage"] = 0.6

    check = coverage_check(summary)

    assert check["n_passing"].tolist() == [3]
    assert check["passes"].tolist() == [False]
    np.testing.assert_allclose(check["lowest_coverage"], [0.6])


def test_coverage_within_one_standard_error_of_the_level_passes():
    close = coverage_check(_summary({5: 0.78}, se=0.03))
    too_far = coverage_check(_summary({5: 0.78}, se=0.01))

    assert close["passes"].tolist() == [True]
    assert too_far["passes"].tolist() == [False]


def test_missing_standard_error_fails():
    summary = _summary({5: 0.95})
    summary["coverage_se"] = np.nan

    assert coverage_check(summary)["passes"].tolist() == [False]


def test_only_rows_at_the_requested_level_are_used():
    both = pd.concat(
        [
            _summary({4: 0.95}, level=0.8),
            _summary({4: 0.3}, level=0.5),
        ],
        ignore_index=True,
    )

    assert coverage_check(both, level=0.8)["passes"].tolist() == [True]
    assert coverage_check(both, level=0.5)["passes"].tolist() == [False]


def test_level_missing_from_the_summary_is_rejected():
    with pytest.raises(ValueError, match="no rows for level"):
        coverage_check(_summary({4: 0.95}), level=0.9)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        coverage_check(_summary({4: 0.95}).drop(columns="coverage_se"))


def test_result_has_one_row_for_each_panel_count_in_order():
    summary = _summary({6: 0.9, 4: 0.95, 5: 0.93})

    check = coverage_check(summary)

    assert check["p"].tolist() == [4, 5, 6]


def test_p0_is_the_smallest_panel_count_that_passes():
    entry = _p0({3: None, 4: 0.95, 5: 0.93, 6: 0.9})

    assert entry == {"status": "reached", "value": 4, "at_grid_minimum": False}


def test_p0_requires_every_larger_panel_count_to_pass():
    entry = _p0({3: None, 4: 0.95, 5: 0.5, 6: 0.95})

    assert entry["value"] == 6


def test_p0_is_not_reached_when_the_largest_panel_count_fails():
    entry = _p0({3: None, 4: 0.95, 5: 0.5})

    assert entry == {"status": "not_reached", "value": None}


def test_p0_at_the_smallest_tested_panel_count_is_marked():
    entry = _p0({3: 0.95, 4: 0.95})

    assert entry == {"status": "reached", "value": 3, "at_grid_minimum": True}


def test_p0_is_not_evaluated_without_interval_results():
    output = derive_thresholds(_degradation_results())

    assert output["thresholds"]["P0"] == {"status": "not_evaluated", "value": None}
    assert output["coverage"]["level"] is None
    assert output["coverage"]["evidence"] == []


def test_p0_uses_the_requested_level():
    summary = pd.concat(
        [
            _summary({3: None, 4: None, 5: 0.5}, level=0.8),
            _summary({3: 0.6, 4: 0.6, 5: 0.6}, level=0.5),
        ],
        ignore_index=True,
    )
    results = _degradation_results()

    at_80 = derive_thresholds(results, intervals=summary)["thresholds"]["P0"]
    at_50 = derive_thresholds(results, intervals=summary, interval_level=0.5)[
        "thresholds"
    ]["P0"]

    assert at_80["status"] == "not_reached"
    assert at_50["value"] == 3


def test_coverage_evidence_and_run_info_are_saved():
    output = derive_thresholds(
        _degradation_results(),
        intervals=_summary({3: None, 4: 0.95}),
        interval_info={"draws": 10},
    )

    coverage = output["coverage"]
    assert coverage["level"] == 0.8
    assert coverage["run"] == {"draws": 10}
    assert [row["p"] for row in coverage["evidence"]] == [3, 4]
    assert set(coverage["evidence"][0]) == {
        "p",
        "n_settings",
        "n_passing",
        "lowest_coverage",
        "passes",
    }
    assert coverage["evidence"][0]["lowest_coverage"] is None
    assert coverage["evidence"][1]["lowest_coverage"] == pytest.approx(0.95)


def test_output_with_intervals_is_strict_json():
    output = derive_thresholds(
        _degradation_results(),
        intervals=_summary({3: None, 4: 0.95}),
    )

    # allow_nan=False raises if any NaN slipped into the output.
    json.dumps(output, allow_nan=False)


def test_interval_results_do_not_change_the_model_thresholds():
    results = _degradation_results()

    without = derive_thresholds(results)["thresholds"]
    with_intervals = derive_thresholds(
        results,
        intervals=_summary({3: None, 4: 0.95}),
    )["thresholds"]

    for key in ("P1", "H1", "W2", "N3", "P3"):
        assert with_intervals[key] == without[key]
