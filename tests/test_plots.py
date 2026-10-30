"""Tests for the prediction card."""

import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from yieldlite.plots import plot_prediction
from yieldlite.result import HistoryRange, Interval, PredictionResult
from yieldlite.selection import Reason


def _prediction(**changes):
    settings = {
        "estimate": 7.41,
        "interval": Interval(6.3, 8.5, 0.8),
        "history": HistoryRange(6.4, 8.4, 0.8, 134),
        "rung": "m1",
        "district": "CD",
        "season": "DX",
    }
    settings.update(changes)
    return PredictionResult(**settings)


def _refusal(**changes):
    settings = {
        "reasons": (
            Reason(
                "interval_uninformative",
                {"width": 2.16, "usual": 1.94},
            ),
        ),
        "history": HistoryRange(6.4, 8.4, 0.8, 134),
        "district": "CD",
        "season": "DX",
    }
    settings.update(changes)
    return PredictionResult(**settings)


def _texts(fig):
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _all_text(fig):
    return "\n".join(_texts(fig))


def test_a_prediction_card_shows_the_estimate_and_both_ranges():
    fig = plot_prediction(_prediction())
    text = _all_text(fig)

    assert isinstance(fig, Figure)
    assert "Expected yield: 7.4 t/ha" in text
    assert "6.3" in text and "8.5" in text
    assert "6.4" in text and "8.4" in text
    assert "Past yields" in text
    assert "CD · DX" in text
    assert "regression (M1)" in text


def test_the_card_has_no_note_section_when_nothing_is_wrong():
    text = _all_text(plot_prediction(_prediction()))

    assert "Please note" not in text
    assert "Why there is no prediction" not in text


def test_a_flagged_card_lists_the_warning_in_plain_words():
    flagged = _prediction(
        reasons=(Reason("variety_pooled", {"variety": "A"}),),
    )
    text = _all_text(plot_prediction(flagged))

    assert "Expected yield" in text
    assert "Please note" in text
    assert "combined with other varieties" in text
    assert "variety_pooled" not in text


def test_a_refused_card_has_no_estimate_and_says_why():
    text = _all_text(plot_prediction(_refusal()))

    assert "No prediction could be made" in text
    assert "Expected yield" not in text
    assert "Why there is no prediction" in text
    assert "2.16" in text
    assert "6.4" in text and "8.4" in text
    assert "interval_uninformative" not in text


def test_a_refusal_without_history_still_draws():
    refused = PredictionResult(
        reasons=(Reason("unknown_district", {"district": "XX"}),),
    )
    fig = plot_prediction(refused)
    text = _all_text(fig)

    assert "No prediction could be made" in text
    assert "XX" in text
    assert fig.axes == []


def test_a_prediction_without_history_says_there_are_too_few_records():
    thin = _prediction(
        history=None,
        reasons=(
            Reason("history_thin", {"district": "CD", "season": "DX"}),
        ),
    )
    text = _all_text(plot_prediction(thin))

    assert "Not enough past records" in text


def test_vietnamese_cards_use_vietnamese_labels():
    text = _all_text(plot_prediction(_prediction(), lang="vi"))

    assert "Năng suất dự kiến: 7.4 t/ha" in text
    assert "Năng suất" in text
    assert "Expected yield" not in text
    assert "Past yields" not in text

    refused = _all_text(plot_prediction(_refusal(), lang="vi"))

    assert "Không thể đưa ra dự đoán" in refused
    assert "Lý do không thể dự đoán" in refused


def test_the_axis_leaves_room_for_every_range():
    fig = plot_prediction(_prediction())
    low, high = fig.axes[0].get_xlim()

    assert low < 6.3 and low < 6.4
    assert high > 8.5 and high > 8.4


def test_a_very_narrow_range_does_not_collapse_the_axis():
    narrow = _prediction(
        estimate=6.0,
        interval=Interval(6.0, 6.0, 0.8),
        history=HistoryRange(6.0, 6.0, 0.8, 20),
    )
    low, high = plot_prediction(narrow).axes[0].get_xlim()

    assert high - low > 1.0


def test_many_warnings_make_a_taller_card():
    short = plot_prediction(_prediction())
    flagged = _prediction(
        reasons=(
            Reason("variety_pooled", {"variety": "A"}),
            Reason(
                "history_thin", {"district": "CD", "season": "DX"}
            ),
        ),
    )

    assert plot_prediction(flagged).get_figheight() > short.get_figheight()


def test_the_card_can_be_saved_as_an_image(tmp_path):
    path = tmp_path / "card.png"

    plot_prediction(_prediction()).savefig(path, dpi=80)

    assert path.stat().st_size > 1000
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_each_call_returns_a_new_figure():
    result = _prediction()

    assert plot_prediction(result) is not plot_prediction(result)


def test_a_bad_language_or_result_is_rejected():
    with pytest.raises(ValueError, match="lang"):
        plot_prediction(_prediction(), lang="fr")

    with pytest.raises(TypeError, match="PredictionResult"):
        plot_prediction({"estimate": 7.0})


def test_the_function_is_available_from_the_package():
    import yieldlite

    assert yieldlite.plot_prediction is plot_prediction
    assert "plot_prediction" in yieldlite.__all__
