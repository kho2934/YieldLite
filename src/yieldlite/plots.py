"""Create a simple visual summary of one yield prediction.

``plot_prediction`` turns a ``PredictionResult`` into a compact prediction
card. The card includes a headline, a chart comparing the prediction interval
with the usual historical yield range, and any warnings or refusal reasons.

Cards can be displayed in English or Vietnamese. The function returns a
standalone Matplotlib ``Figure``, so it works in scripts, notebooks, and
server environments without relying on ``pyplot``.

Example
-------
fig = plot_prediction(result, lang="vi")
fig.savefig("prediction_card.png", dpi=200, bbox_inches="tight")
"""

from __future__ import annotations

import textwrap

from matplotlib.figure import Figure

from yieldlite.messages import (
    LANGUAGES,
    RUNG_NAMES,
    render_reasons,
)
from yieldlite.result import PredictionResult

# The prediction and historical ranges use colors that remain distinguishable
# for most common forms of color-vision deficiency.
MODEL_COLOR = "#1b5fa6"
HISTORY_COLOR = "#8c9199"

# Text colors used throughout the card.
INK = "#1f2328"
MUTED = "#5b626b"
ALERT = "#a34a00"

_WIDTH_IN = 8.0
_WRAP = 88

LABELS: dict[str, dict[str, str]] = {
    "title": {
        "en": "Yield prediction",
        "vi": "Dự đoán năng suất lúa",
    },
    "headline": {
        "en": "Expected yield: {estimate:.1f} t/ha",
        "vi": "Năng suất dự kiến: {estimate:.1f} t/ha",
    },
    "subtitle": {
        "en": (
            "Most likely between {lower:.1f} and {upper:.1f} t/ha "
            "({level}% range)"
        ),
        "vi": (
            "Nhiều khả năng nằm trong khoảng từ {lower:.1f} đến "
            "{upper:.1f} t/ha (khoảng dự đoán {level}%)"
        ),
    },
    "refused": {
        "en": "No prediction could be made",
        "vi": "Không thể đưa ra dự đoán",
    },
    "row_prediction": {
        "en": "Prediction",
        "vi": "Dự đoán",
    },
    "row_history": {
        "en": "Past yields",
        "vi": "Năng suất\ntrước đây",
    },
    "no_prediction": {
        "en": "No prediction",
        "vi": "Không có kết quả dự đoán",
    },
    "no_history": {
        "en": "Not enough past records",
        "vi": "Không đủ dữ liệu lịch sử",
    },
    "axis": {
        "en": "Yield (t/ha)",
        "vi": "Năng suất (t/ha)",
    },
    "notes_flagged": {
        "en": "Please note",
        "vi": "Lưu ý",
    },
    "notes_refused": {
        "en": "Why there is no prediction",
        "vi": "Lý do không thể dự đoán",
    },
    "model": {
        "en": "Model: {name}",
        "vi": "Mô hình: {name}",
    },
}


def _label(
    key: str,
    lang: str,
    **params: object,
) -> str:
    """Return a translated label with its values filled in."""
    return LABELS[key][lang].format(**params)


def _model_name(
    rung: str,
    lang: str,
) -> str:
    """Return the readable name of a model with a capital first letter."""
    name = RUNG_NAMES[rung][lang]

    return name[:1].upper() + name[1:]


def _percent(level: float) -> int:
    """Convert a level such as 0.8 to a whole percentage."""
    return int(round(level * 100))


def _wrapped(lines: list[str]) -> list[str]:
    """Wrap long explanations into bullets that fit inside the card."""
    wrapped: list[str] = []

    for line in lines:
        wrapped.extend(
            textwrap.wrap(
                line,
                width=_WRAP,
                initial_indent="• ",
                subsequent_indent="  ",
            )
        )

    return wrapped


def _x_limits(
    result: PredictionResult,
) -> tuple[float, float]:
    """Choose readable horizontal limits for the range chart."""
    values: list[float] = []

    if result.interval is not None:
        values.extend(
            [
                result.interval.lower,
                result.interval.upper,
            ]
        )

    if result.history is not None:
        values.extend(
            [
                result.history.lower,
                result.history.upper,
            ]
        )

    low = min(values)
    high = max(values)
    middle = (low + high) / 2

    # Keep at least one t/ha of visible space and add room for the endpoint
    # labels on both sides of the ranges.
    half_width = (
        max(high - low, 1.0)
        / 2
        * 1.36
    )

    return (
        middle - half_width,
        middle + half_width,
    )


def _draw_bar(
    ax,
    y: float,
    lower: float,
    upper: float,
    color: str,
) -> None:
    """Draw one yield range and label both ends."""
    ax.barh(
        y,
        upper - lower,
        left=lower,
        height=0.34,
        color=color,
        alpha=0.9,
        zorder=2,
    )

    span = (
        ax.get_xlim()[1]
        - ax.get_xlim()[0]
    )

    endpoints = (
        (lower, "right", -0.03),
        (upper, "left", 0.03),
    )

    for value, alignment, offset in endpoints:
        ax.text(
            value + offset * span,
            y,
            f"{value:.1f}",
            ha=alignment,
            va="center",
            fontsize=10,
            color=INK,
        )


def plot_prediction(
    result: PredictionResult,
    lang: str = "en",
) -> Figure:
    """Draw one prediction result as a reader-friendly card.

    Parameters
    ----------
    result : PredictionResult
        Result returned by ``YieldModel.predict``.
    lang : {"en", "vi"}, default "en"
        Language used for all labels and explanations.

    Returns
    -------
    matplotlib.figure.Figure
        Completed prediction card. A successful prediction shows the model
        interval above the historical range. A refusal shows the historical
        range when one is available and explains why no prediction was made.
    """
    if not isinstance(result, PredictionResult):
        raise TypeError(
            "result must be a PredictionResult"
        )

    if lang not in LANGUAGES:
        raise ValueError(
            f"lang must be one of {list(LANGUAGES)}"
        )

    refused = result.refused
    notes = _wrapped(
        render_reasons(
            result.reasons,
            lang,
        )
    )

    has_chart = (
        result.interval is not None
        or result.history is not None
    )

    # Work out the height from the content so cards with several warnings have
    # enough room without leaving excessive blank space.
    top_in = 1.0
    chart_in = 1.9 if has_chart else 0.0
    axis_in = 0.8 if has_chart else 0.0
    notes_in = (
        0.3 + 0.24 * len(notes)
        if notes
        else 0.0
    )
    footer_in = 0.45

    height = (
        top_in
        + chart_in
        + axis_in
        + notes_in
        + footer_in
    )

    fig = Figure(
        figsize=(_WIDTH_IN, height),
        facecolor="white",
    )

    def y_of(
        inches_from_top: float,
    ) -> float:
        """Convert a distance from the top into figure coordinates."""
        return 1 - inches_from_top / height

    title = _label(
        "title",
        lang,
    )

    if result.district and result.season:
        title += (
            f" · {result.district}"
            f" · {result.season}"
        )

    fig.text(
        0.04,
        y_of(0.35),
        title,
        fontsize=11,
        color=MUTED,
        va="center",
    )

    if refused:
        fig.text(
            0.04,
            y_of(0.72),
            _label(
                "refused",
                lang,
            ),
            fontsize=17,
            fontweight="bold",
            color=ALERT,
            va="center",
        )
    else:
        fig.text(
            0.04,
            y_of(0.68),
            _label(
                "headline",
                lang,
                estimate=result.estimate,
            ),
            fontsize=18,
            fontweight="bold",
            color=INK,
            va="center",
        )

        fig.text(
            0.04,
            y_of(0.95),
            _label(
                "subtitle",
                lang,
                lower=result.interval.lower,
                upper=result.interval.upper,
                level=_percent(
                    result.interval.level
                ),
            ),
            fontsize=11,
            color=MUTED,
            va="center",
        )

    if has_chart:
        left = 0.20
        right = 0.96
        bottom = y_of(
            top_in
            + chart_in
            + axis_in * 0.55
        )

        ax = fig.add_axes(
            [
                left,
                bottom,
                right - left,
                chart_in / height,
            ]
        )

        ax.set_xlim(
            *_x_limits(result)
        )
        ax.set_ylim(-0.6, 1.6)

        for side in (
            "top",
            "right",
            "left",
        ):
            ax.spines[side].set_visible(False)

        ax.spines["bottom"].set_color(
            "#c9ced4"
        )
        ax.tick_params(
            axis="x",
            colors=MUTED,
            labelsize=10,
        )
        ax.tick_params(
            axis="y",
            length=0,
        )
        ax.grid(
            axis="x",
            color="#e6e9ec",
            linewidth=0.8,
            zorder=0,
        )
        ax.set_axisbelow(True)
        ax.set_xlabel(
            _label(
                "axis",
                lang,
            ),
            fontsize=10,
            color=MUTED,
        )
        ax.set_yticks([1, 0])
        ax.set_yticklabels(
            [
                _label(
                    "row_prediction",
                    lang,
                ),
                _label(
                    "row_history",
                    lang,
                ),
            ],
            fontsize=11,
            color=INK,
        )

        if result.interval is not None:
            _draw_bar(
                ax,
                1,
                result.interval.lower,
                result.interval.upper,
                MODEL_COLOR,
            )

            # The open circle marks the model's point estimate.
            ax.plot(
                [result.estimate],
                [1],
                marker="o",
                markersize=11,
                markerfacecolor="white",
                markeredgecolor=INK,
                markeredgewidth=2,
                zorder=3,
            )
        else:
            middle = sum(
                ax.get_xlim()
            ) / 2

            ax.text(
                middle,
                1,
                _label(
                    "no_prediction",
                    lang,
                ),
                ha="center",
                va="center",
                fontsize=11,
                color=ALERT,
            )

        if result.history is not None:
            _draw_bar(
                ax,
                0,
                result.history.lower,
                result.history.upper,
                HISTORY_COLOR,
            )
        else:
            middle = sum(
                ax.get_xlim()
            ) / 2

            ax.text(
                middle,
                0,
                _label(
                    "no_history",
                    lang,
                ),
                ha="center",
                va="center",
                fontsize=11,
                color=MUTED,
            )

    if notes:
        heading = (
            "notes_refused"
            if refused
            else "notes_flagged"
        )
        start = (
            top_in
            + chart_in
            + axis_in
            + 0.15
        )

        fig.text(
            0.04,
            y_of(start),
            _label(
                heading,
                lang,
            ),
            fontsize=11,
            fontweight="bold",
            color=(
                ALERT
                if refused
                else INK
            ),
            va="center",
        )

        for index, line in enumerate(notes):
            fig.text(
                0.04,
                y_of(
                    start
                    + 0.3
                    + 0.24 * index
                ),
                line,
                fontsize=10.5,
                color=INK,
                va="center",
            )

    if result.rung is not None:
        fig.text(
            0.04,
            y_of(height - 0.2),
            _label(
                "model",
                lang,
                name=_model_name(
                    result.rung,
                    lang,
                ),
            ),
            fontsize=9,
            color=MUTED,
            va="center",
        )

    return fig
