"""Tests for the model selection policy."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from yieldlite.profile import DataProfile
from yieldlite.selection import (
    THRESHOLD_NAMES,
    LeaveOnePanelOutGain,
    PanelGain,
    Reason,
    SelectionPolicy,
)

_WIN = PanelGain(mean=0.04, se=0.01, n_panels=8)
_TIE = PanelGain(mean=0.01, se=0.01, n_panels=8)
_UNKNOWN = PanelGain(mean=math.nan, se=math.nan, n_panels=0)


def _reached(value):
    return {"status": "reached", "value": value}


def _not(status):
    return {"status": status, "value": None}


def _document(**changes):
    """Thresholds as the studies wrote them: M1 is supported, the rest is not."""
    entries = {
        "P0": _reached(4),
        "P1": _reached(3),
        "H1": _reached(20),
        "W2": _not("not_evaluated"),
        "N3": _not("not_evaluated"),
        "P3": _not("not_evaluated"),
    }
    entries.update(changes)
    return {"thresholds": entries}


def _profile(**changes):
    """The eight development panels."""
    profile = DataProfile(
        n_rows=1327,
        n_panels=8,
        n_seasons=3,
        panels_per_season={"DX": 3, "HT": 2, "TD": 3},
        median_households_per_cell=40.0,
        within_season_contrasts=5,
        rows_per_variety={"pooled": 100},
        flagged_share=0.0,
    )
    return dataclasses.replace(profile, **changes)


class _FakeGain:
    """Return prepared gains and remember which comparisons were asked for."""

    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def __call__(self, better, baseline):
        self.calls.append((better, baseline))
        return self.results[(better, baseline)]


def _codes(selection):
    return [reason.code for reason in selection.reasons]


def test_packaged_thresholds_load():
    policy = SelectionPolicy()

    assert set(policy.thresholds) == set(THRESHOLD_NAMES)


def test_too_few_panels_refuses_without_comparing_rungs():
    gain = _FakeGain()

    selection = SelectionPolicy(_document()).select(_profile(n_panels=3), gain)

    assert selection.refused
    assert selection.rung is None
    assert selection.reasons == (Reason("too_few_panels", {"panels": 3, "needed": 4}),)
    assert gain.calls == []


@pytest.mark.parametrize("status", ["not_reached", "not_evaluated"])
def test_unreached_p0_refuses_however_many_panels_there_are(status):
    policy = SelectionPolicy(_document(P0=_not(status)))

    selection = policy.select(_profile(n_panels=50), _FakeGain())

    assert selection.refused
    assert selection.reasons == (
        Reason("too_few_panels", {"panels": 50, "needed": None}),
    )


def test_exactly_p0_panels_is_enough_to_fit():
    gain = _FakeGain({("m1", "m0"): _TIE})

    selection = SelectionPolicy(_document()).select(_profile(n_panels=4), gain)

    assert not selection.refused
    assert selection.rung == "m0"


def test_m1_is_chosen_when_the_data_allow_it_and_it_beats_m0():
    gain = _FakeGain({("m1", "m0"): _WIN})

    selection = SelectionPolicy(_document()).select(_profile(), gain)

    assert selection.rung == "m1"
    assert selection.eligible == ("m0", "m1")
    assert selection.gains == {"m1": _WIN}
    assert gain.calls == [("m1", "m0")]

    # The rungs above are reported first, then the rung that was chosen.
    assert selection.reasons == (
        Reason("rung_not_evaluated", {"rung": "m3"}),
        Reason("rung_not_evaluated", {"rung": "m2"}),
        Reason(
            "rung_selected",
            {
                "rung": "m1",
                "panels": 8,
                "simpler": "m0",
                "gain": 0.04,
                "se": 0.01,
                "compared": 8,
            },
        ),
    )


def test_m0_is_chosen_when_m1_shows_no_gain():
    gain = _FakeGain({("m1", "m0"): _TIE})

    selection = SelectionPolicy(_document()).select(_profile(), gain)

    assert selection.rung == "m0"
    assert selection.eligible == ("m0", "m1")
    assert selection.reasons[2:] == (
        Reason(
            "no_gain",
            {
                "rung": "m1",
                "simpler": "m0",
                "compared": 8,
                "gain": 0.01,
                "se": 0.01,
            },
        ),
        Reason("rung_selected", {"rung": "m0", "panels": 8}),
    )


@pytest.mark.parametrize(
    "mean, rung",
    [(0.011, "m1"), (0.010, "m0"), (0.005, "m0"), (-0.02, "m0")],
)
def test_a_gain_has_to_be_larger_than_its_standard_error(mean, rung):
    gain = _FakeGain({("m1", "m0"): PanelGain(mean=mean, se=0.01, n_panels=8)})

    selection = SelectionPolicy(_document()).select(_profile(), gain)

    assert selection.rung == rung


@pytest.mark.parametrize(
    "result",
    [
        _UNKNOWN,
        PanelGain(mean=0.05, se=math.nan, n_panels=1),
        PanelGain(mean=0.05, se=0.01, n_panels=1),
    ],
)
def test_a_gain_that_cannot_be_assessed_steps_down(result):
    gain = _FakeGain({("m1", "m0"): result})

    selection = SelectionPolicy(_document()).select(_profile(), gain)

    assert selection.rung == "m0"
    assert Reason(
        "gain_not_assessable",
        {"rung": "m1", "simpler": "m0", "compared": result.n_panels},
    ) in selection.reasons


@pytest.mark.parametrize(
    "mean, se, n_panels, expected",
    [
        (0.04, 0.01, 8, True),
        (0.01, 0.01, 8, False),
        (0.04, 0.01, 1, False),
        (math.nan, 0.01, 8, False),
        (0.04, math.nan, 8, False),
    ],
)
def test_panel_gain_shows_a_gain_only_when_it_beats_its_standard_error(
    mean, se, n_panels, expected
):
    assert PanelGain(mean, se, n_panels).shows_gain is expected


def test_too_few_panels_for_m1_blocks_it_without_a_comparison():
    gain = _FakeGain()

    selection = SelectionPolicy(_document(P1=_reached(5))).select(
        _profile(n_panels=4), gain
    )

    assert selection.rung == "m0"
    assert selection.eligible == ("m0",)
    assert Reason(
        "rung_needs_more_panels", {"rung": "m1", "have": 4, "needed": 5}
    ) in selection.reasons
    assert gain.calls == []


def test_too_few_households_per_cell_blocks_m1():
    selection = SelectionPolicy(_document()).select(
        _profile(median_households_per_cell=12.5), _FakeGain()
    )

    assert selection.rung == "m0"
    assert Reason(
        "rung_needs_more_households", {"rung": "m1", "have": 12.5, "needed": 20}
    ) in selection.reasons


def test_every_unmet_requirement_is_reported():
    policy = SelectionPolicy(_document(P1=_reached(6)))

    selection = policy.select(
        _profile(n_panels=5, median_households_per_cell=10.0), _FakeGain()
    )

    m1_codes = [r.code for r in selection.reasons if r.params.get("rung") == "m1"]
    assert m1_codes == ["rung_needs_more_panels", "rung_needs_more_households"]


@pytest.mark.parametrize(
    "name, status, code",
    [
        ("P1", "not_reached", "rung_not_reached"),
        ("H1", "not_reached", "rung_not_reached"),
        ("P1", "not_evaluated", "rung_not_evaluated"),
        ("H1", "not_evaluated", "rung_not_evaluated"),
    ],
)
def test_a_threshold_the_studies_did_not_reach_blocks_the_rung(name, status, code):
    policy = SelectionPolicy(_document(**{name: _not(status)}))

    selection = policy.select(_profile(), _FakeGain())

    assert selection.rung == "m0"
    assert Reason(code, {"rung": "m1"}) in selection.reasons


def test_not_reached_outranks_not_evaluated():
    policy = SelectionPolicy(
        _document(P1=_not("not_evaluated"), H1=_not("not_reached"))
    )

    selection = policy.select(_profile(), _FakeGain())

    assert Reason("rung_not_reached", {"rung": "m1"}) in selection.reasons
    assert Reason("rung_not_evaluated", {"rung": "m1"}) not in selection.reasons


def test_m2_needs_enough_weather_contrasts():
    policy = SelectionPolicy(_document(W2=_reached(10)))

    selection = policy.select(_profile(), _FakeGain({("m1", "m0"): _WIN}))

    assert selection.rung == "m1"
    assert Reason(
        "rung_needs_more_contrasts", {"rung": "m2", "have": 5, "needed": 10}
    ) in selection.reasons


def test_m2_is_compared_with_m1_and_steps_down_when_it_does_not_win():
    policy = SelectionPolicy(_document(W2=_reached(5)))
    gain = _FakeGain({("m2", "m1"): _TIE, ("m1", "m0"): _WIN})

    selection = policy.select(_profile(), gain)

    assert selection.rung == "m1"
    assert selection.eligible == ("m0", "m1", "m2")
    assert gain.calls == [("m2", "m1"), ("m1", "m0")]
    assert _codes(selection) == [
        "rung_not_evaluated",
        "no_gain",
        "rung_selected",
    ]


def test_the_search_stops_at_the_first_rung_that_wins():
    policy = SelectionPolicy(_document(W2=_reached(5)))
    gain = _FakeGain({("m2", "m1"): _WIN})

    selection = policy.select(_profile(), gain)

    assert selection.rung == "m2"
    assert gain.calls == [("m2", "m1")]


def test_m2_is_blocked_when_m1_is_not_eligible():
    policy = SelectionPolicy(_document(P1=_reached(9), W2=_reached(5)))

    selection = policy.select(_profile(), _FakeGain())

    assert selection.rung == "m0"
    assert _codes(selection) == [
        "rung_not_evaluated",
        "rung_blocked",
        "rung_needs_more_panels",
        "rung_selected",
    ]
    assert selection.reasons[1].params == {"rung": "m2", "needs": "m1"}


def test_m3_needs_both_rows_and_panels():
    policy = SelectionPolicy(_document(N3=_reached(2000), P3=_reached(10)))

    selection = policy.select(_profile(), _FakeGain({("m1", "m0"): _WIN}))

    assert selection.rung == "m1"
    assert selection.reasons[:2] == (
        Reason("rung_needs_more_rows", {"rung": "m3", "have": 1327, "needed": 2000}),
        Reason("rung_needs_more_panels", {"rung": "m3", "have": 8, "needed": 10}),
    )


def test_m3_is_compared_with_the_next_simpler_rung_that_is_eligible():
    policy = SelectionPolicy(_document(N3=_reached(1000), P3=_reached(6)))
    gain = _FakeGain({("m3", "m1"): _WIN})

    selection = policy.select(_profile(), gain)

    assert selection.rung == "m3"
    assert selection.eligible == ("m0", "m1", "m3")
    assert gain.calls == [("m3", "m1")]


def test_a_gain_function_is_required_when_a_higher_rung_is_eligible():
    with pytest.raises(ValueError, match="gain function"):
        SelectionPolicy(_document()).select(_profile())


def test_no_gain_function_is_needed_when_only_m0_is_eligible():
    policy = SelectionPolicy(_document(P1=_reached(99)))

    selection = policy.select(_profile())

    assert selection.rung == "m0"
    assert selection.gains == {}


def test_profile_must_be_a_data_profile():
    with pytest.raises(TypeError, match="DataProfile"):
        SelectionPolicy(_document()).select({"n_panels": 8})


@pytest.mark.parametrize(
    "document",
    ["text", {}, {"thresholds": None}, {"thresholds": {}}],
)
def test_a_malformed_document_is_rejected(document):
    with pytest.raises(ValueError):
        SelectionPolicy(document)


def test_a_missing_threshold_is_rejected():
    document = _document()
    del document["thresholds"]["H1"]

    with pytest.raises(ValueError, match="missing H1"):
        SelectionPolicy(document)


def test_an_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="status"):
        SelectionPolicy(_document(P1={"status": "maybe", "value": 3}))


@pytest.mark.parametrize("value", [None, "all", True, -1, math.inf, math.nan])
def test_a_reached_threshold_needs_a_number(value):
    with pytest.raises(ValueError, match="not a non-negative number"):
        SelectionPolicy(_document(H1=_reached(value)))


def test_whole_number_thresholds_are_reported_as_integers():
    policy = SelectionPolicy(_document(P1=_reached(5.0)))

    selection = policy.select(_profile(n_panels=4), _FakeGain())

    reason = next(r for r in selection.reasons if r.code == "rung_needs_more_panels")
    assert reason.params["needed"] == 5
    assert isinstance(reason.params["needed"], int)


# Leave-one-panel-out gain on a table of households.

_PANELS = {
    "DX_2020": "DX",
    "HT_2020": "HT",
    "TD_2020": "TD",
    "DX_2021": "DX",
    "TD_2021": "TD",
    "DX_2022": "DX",
    "HT_2022": "HT",
    "TD_2022": "TD",
}
_DISTRICTS = ["CD", "OM", "TL", "TN", "VT"]
_SEASON_EFFECT = {"DX": 6.5, "HT": 5.0, "TD": 4.5}


def _frame(panels=None, district_effect=0.2, fertilizer_effect=0.04, households=12):
    """Households whose yield depends on season, district and fertilizer."""
    rng = np.random.default_rng(1)
    rows = []
    for panel, season in (panels or _PANELS).items():
        for k, district in enumerate(_DISTRICTS):
            for i in range(households):
                npk = 120.0 + 10.0 * (i + k)
                rows.append(
                    {
                        "panel_id": panel,
                        "season": season,
                        "district_code": district,
                        "variety_raw": "A" if i % 2 == 0 else "B",
                        "variety_flag": "known",
                        "npk_total_kg_ha": npk,
                        "yield_t_ha": _SEASON_EFFECT[season]
                        + district_effect * k
                        + fertilizer_effect * (npk - 170.0)
                        + rng.normal(0, 0.2),
                    }
                )
    return pd.DataFrame(rows)


def test_inputs_that_explain_yield_give_m1_a_clear_gain():
    gain = LeaveOnePanelOutGain(_frame(), min_variety_count=5)("m1", "m0")

    assert gain.n_panels == len(_PANELS)
    assert gain.mean > 0.1
    assert gain.shows_gain


def test_inputs_that_explain_nothing_give_m1_no_clear_gain():
    # Yield differs by season only, so the extra inputs have nothing to find.
    frame = _frame(district_effect=0.0, fertilizer_effect=0.0)

    gain = LeaveOnePanelOutGain(frame, min_variety_count=5)("m1", "m0")

    assert gain.n_panels == len(_PANELS)
    assert abs(gain.mean) < 0.01
    assert not gain.shows_gain


def test_a_rung_gains_nothing_over_itself():
    gain = LeaveOnePanelOutGain(_frame(), min_variety_count=5)("m1", "m1")

    assert gain.mean == 0.0
    assert gain.se == 0.0
    assert not gain.shows_gain


def test_a_season_with_one_panel_is_left_out_of_the_comparison():
    panels = {"DX_2020": "DX", "DX_2021": "DX", "HT_2020": "HT"}

    comparison = LeaveOnePanelOutGain(_frame(panels), min_variety_count=5)

    assert comparison.comparable_panels == ["DX_2020", "DX_2021"]
    assert comparison("m1", "m0").n_panels == 2


def test_one_panel_per_season_leaves_nothing_to_compare():
    panels = {"DX_2020": "DX", "HT_2020": "HT", "TD_2020": "TD"}

    comparison = LeaveOnePanelOutGain(_frame(panels), min_variety_count=5)
    gain = comparison("m1", "m0")

    assert comparison.comparable_panels == []
    assert gain.n_panels == 0
    assert math.isnan(gain.mean)
    assert not gain.assessable


def test_the_frame_is_not_changed():
    frame = _frame()
    before = frame.copy()

    LeaveOnePanelOutGain(frame, min_variety_count=5)("m1", "m0")

    pd.testing.assert_frame_equal(frame, before)


def test_an_unknown_rung_is_rejected():
    comparison = LeaveOnePanelOutGain(_frame(), min_variety_count=5)

    with pytest.raises(ValueError, match="unknown rung"):
        comparison("m9", "m0")


def test_the_frame_must_be_a_dataframe_with_panels_and_seasons():
    with pytest.raises(TypeError):
        LeaveOnePanelOutGain(_frame().to_numpy())

    with pytest.raises(ValueError, match="missing"):
        LeaveOnePanelOutGain(_frame().drop(columns="season"))


def test_the_policy_picks_m1_on_data_where_the_inputs_matter():
    frame = _frame()
    frame["variety_pooled"] = frame["variety_raw"]
    frame["flagged"] = False
    profile = DataProfile.from_frame(frame)

    policy = SelectionPolicy(_document())
    selection = policy.select(profile, LeaveOnePanelOutGain(frame, min_variety_count=5))

    assert selection.rung == "m1"
    assert selection.gains["m1"].shows_gain


def test_the_policy_keeps_m0_on_data_where_only_the_season_matters():
    frame = _frame(district_effect=0.0, fertilizer_effect=0.0)
    frame["variety_pooled"] = frame["variety_raw"]
    frame["flagged"] = False
    profile = DataProfile.from_frame(frame)

    policy = SelectionPolicy(_document())
    selection = policy.select(profile, LeaveOnePanelOutGain(frame, min_variety_count=5))

    assert selection.rung == "m0"
