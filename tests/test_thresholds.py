"""Tests for turning degradation-study results into selection thresholds."""

import json

import numpy as np
import pandas as pd
import pytest

from yieldlite.thresholds import (
    derive_thresholds,
    paired_gain,
    write_thresholds,
)

_PANELS = list("ABCDEFGH")
_OFFSETS = np.linspace(-0.01, 0.01, len(_PANELS))


def _results(
    gain_by_setting: dict[tuple[int, float], float],
) -> pd.DataFrame:
    """Create test results with a chosen M1 improvement at each setting."""
    rows = []

    for (p, h), gain in gain_by_setting.items():
        for panel, offset in zip(_PANELS, _OFFSETS, strict=True):
            rows.append((panel, p, h, "m0", 1.0))
            rows.append((panel, p, h, "m1", 1.0 - gain - offset))

    return pd.DataFrame(
        rows,
        columns=["held_out", "p", "h", "model", "mae"],
    )


def _grid(
    p_gain: dict[int, float],
    h_gain: dict[float, float],
) -> pd.DataFrame:
    """Create a small panel-count and household-cap test grid."""
    settings = {(p, np.inf): gain for p, gain in p_gain.items()}
    largest_panel_count = max(p_gain)

    settings.update(
        {
            (largest_panel_count, h): gain
            for h, gain in h_gain.items()
        }
    )

    return _results(settings)


def test_paired_gain_matches_hand_calculation():
    results = pd.DataFrame(
        {
            "held_out": ["A", "A", "B", "B", "C", "C"],
            "p": [3] * 6,
            "h": [np.inf] * 6,
            "model": ["m0", "m1"] * 3,
            "mae": [1.0, 0.9, 1.0, 0.8, 1.0, 0.7],
        }
    )

    gains = paired_gain(results)

    row = gains.iloc[0]
    assert row["mean_gain"] == pytest.approx(0.2)
    assert row["se"] == pytest.approx(0.1 / np.sqrt(3))
    assert row["n_panels"] == 3
    assert bool(row["passes"])


def test_gain_of_zero_does_not_pass():
    gains = paired_gain(_results({(3, np.inf): 0.0}))

    assert not bool(gains["passes"].iloc[0])


def test_single_panel_cannot_pass():
    results = pd.DataFrame(
        {
            "held_out": ["A", "A"],
            "p": [3, 3],
            "h": [np.inf] * 2,
            "model": ["m0", "m1"],
            "mae": [1.0, 0.5],
        }
    )

    gains = paired_gain(results)

    assert not bool(gains["passes"].iloc[0])


def test_missing_model_is_rejected():
    results = _results({(3, np.inf): 0.1})

    with pytest.raises(ValueError):
        paired_gain(results[results["model"] == "m0"])


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError):
        paired_gain(pd.DataFrame({"mae": [1.0]}))


def test_p1_is_the_first_setting_that_passes():
    results = _grid(
        {3: 0.0, 4: 0.0, 5: 0.1, 6: 0.1, 7: 0.1},
        {5.0: 0.0, np.inf: 0.1},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["P1"] == {
        "status": "reached",
        "value": 5,
        "at_grid_minimum": False,
    }


def test_p1_requires_every_larger_setting_to_pass():
    results = _grid(
        {3: 0.0, 4: 0.0, 5: 0.1, 6: 0.0, 7: 0.1},
        {5.0: 0.0, np.inf: 0.1},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["P1"]["value"] == 7


def test_unreached_crossover_is_recorded_not_guessed():
    results = _grid(
        {3: 0.0, 4: 0.0, 5: 0.0},
        {5.0: 0.0, np.inf: 0.0},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["P1"] == {
        "status": "not_reached",
        "value": None,
    }
    assert thresholds["H1"] == {
        "status": "not_reached",
        "value": None,
    }


def test_h1_uses_the_largest_panel_count_and_labels_no_cap_as_all():
    results = _grid(
        {3: 0.0, 4: 0.1, 5: 0.1},
        {5.0: 0.0, 10.0: 0.0, np.inf: 0.1},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["H1"] == {
        "status": "reached",
        "value": "all",
        "at_grid_minimum": False,
    }


def test_h1_returns_a_number_when_a_cap_is_enough():
    results = _grid(
        {3: 0.0, 4: 0.1, 5: 0.1},
        {5.0: 0.0, 10.0: 0.1, np.inf: 0.1},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["H1"] == {
        "status": "reached",
        "value": 10,
        "at_grid_minimum": False,
    }


def test_p1_is_not_evaluated_without_an_uncapped_run():
    results = _results(
        {
            (3, 5.0): 0.1,
            (4, 5.0): 0.1,
        }
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["P1"] == {
        "status": "not_evaluated",
        "value": None,
    }


def test_unbuilt_rungs_and_coverage_are_not_evaluated():
    results = _grid(
        {3: 0.1, 4: 0.1},
        {5.0: 0.1, np.inf: 0.1},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    for key in ("P0", "W2", "N3", "P3"):
        assert thresholds[key] == {
            "status": "not_evaluated",
            "value": None,
        }


def test_output_carries_evidence_and_run_info():
    results = _grid(
        {3: 0.1, 4: 0.1},
        {5.0: 0.1, np.inf: 0.1},
    )

    output = derive_thresholds(
        results,
        run_info={"seed": 0},
    )

    assert output["n_development_panels"] == len(_PANELS)
    assert output["run"] == {"seed": 0}
    assert output["evidence"]
    assert {
        "p",
        "h",
        "mean_gain",
        "se",
        "n_panels",
        "passes",
    } <= set(output["evidence"][0])


def test_written_file_is_valid_json(tmp_path):
    results = _grid(
        {3: 0.1, 4: 0.1},
        {5.0: 0.1, np.inf: 0.1},
    )
    target = tmp_path / "data" / "thresholds.json"

    write_thresholds(
        derive_thresholds(results),
        target,
    )

    loaded = json.loads(target.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert "P1" in loaded["thresholds"]


def _joint_grid() -> pd.DataFrame:
    """Create a grid where small panel counts need more households."""
    gain = {
        (3, 5.0): 0.0,
        (3, 10.0): 0.0,
        (3, 20.0): 0.1,
        (3, np.inf): 0.1,
        (4, 5.0): 0.0,
        (4, 10.0): 0.1,
        (4, 20.0): 0.1,
        (4, np.inf): 0.1,
        (5, 5.0): 0.1,
        (5, 10.0): 0.1,
        (5, 20.0): 0.1,
        (5, np.inf): 0.1,
    }
    return _results(gain)


def test_h1_holds_at_every_panel_count_from_p1_upward():
    thresholds = derive_thresholds(_joint_grid())["thresholds"]

    assert thresholds["P1"]["value"] == 3
    assert thresholds["H1"]["value"] == 20


def test_every_tested_setting_inside_the_region_passes():
    output = derive_thresholds(_joint_grid())
    p1 = output["thresholds"]["P1"]["value"]
    h1 = output["thresholds"]["H1"]["value"]

    inside = [
        row
        for row in output["evidence"]
        if row["p"] >= p1 and (row["h"] == "all" or row["h"] >= h1)
    ]

    assert inside
    assert all(row["passes"] for row in inside)


def test_threshold_at_the_smallest_tested_value_is_marked():
    results = _grid(
        {3: 0.1, 4: 0.1, 5: 0.1},
        {5.0: 0.1, np.inf: 0.1},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["P1"]["at_grid_minimum"] is True
    assert thresholds["H1"]["at_grid_minimum"] is True


def test_h1_is_not_reached_when_p1_is_not_reached():
    results = _grid(
        {3: 0.0, 4: 0.0, 5: 0.0},
        {5.0: 0.1, np.inf: 0.0},
    )

    thresholds = derive_thresholds(results)["thresholds"]

    assert thresholds["P1"]["status"] == "not_reached"
    assert thresholds["H1"]["status"] == "not_reached"
