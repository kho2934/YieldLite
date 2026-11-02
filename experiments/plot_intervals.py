"""Turn the interval-study results into one comparison figure.

Example
-------
python experiments/plot_intervals.py

The script reads ``interval_summary.csv`` and saves ``intervals.png`` in the
same results folder. Both panels show M1 at the 80% level with no household
cap. This keeps everything fixed except the number of training panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"
M1_COLOR = "#eb6834"
HISTORY_COLOR = "#3b6ea5"

_REQUIRED_COLUMNS = {
    "model",
    "h",
    "p",
    "level",
    "supported",
    "coverage",
    "coverage_se",
    "history_coverage",
    "history_coverage_se",
    "width",
    "ref_width",
}


def _read_boolean(values: pd.Series, name: str) -> pd.Series:
    """Read a CSV column containing True/False or 1/0 values."""
    if values.isna().any():
        raise ValueError(f"{name} cannot contain missing values")

    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)

    text = values.astype(str).str.strip().str.lower()
    valid = text.isin({"true", "false", "1", "0"})

    if not valid.all():
        unexpected = sorted(text.loc[~valid].unique())
        raise ValueError(
            f"{name} must contain True/False or 1/0; found {unexpected}"
        )

    return text.isin({"true", "1"})


def _select(summary: pd.DataFrame, level: float) -> pd.DataFrame:
    """Select the uncapped M1 results for one interval level."""
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")

    missing = sorted(_REQUIRED_COLUMNS - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be a finite number between 0 and 1")

    summary = summary.copy()

    # The results normally store an uncapped run as infinity. Accept "all" as
    # well so the plotting script can also read a manually edited CSV.
    all_rows = summary["h"].astype(str).str.strip().str.lower().eq("all")
    summary.loc[all_rows, "h"] = np.inf

    numeric_columns = [
        "h",
        "p",
        "level",
        "coverage",
        "coverage_se",
        "history_coverage",
        "history_coverage_se",
        "width",
        "ref_width",
    ]
    for column in numeric_columns:
        summary[column] = pd.to_numeric(summary[column], errors="raise")

    summary["supported"] = _read_boolean(
        summary["supported"],
        name="supported",
    )

    chosen = summary.loc[
        (summary["model"] == "m1")
        & np.isclose(summary["level"], level)
        & np.isinf(summary["h"])
    ].copy()

    if chosen.empty:
        raise ValueError(
            "no uncapped M1 results were found at the requested level"
        )

    if not np.allclose(chosen["p"], np.round(chosen["p"])):
        raise ValueError("training-panel counts must be whole numbers")

    chosen["p"] = chosen["p"].astype(int)

    if chosen["p"].duplicated().any():
        duplicates = sorted(chosen.loc[chosen["p"].duplicated(), "p"].unique())
        raise ValueError(
            f"more than one result was found for panel counts {duplicates}"
        )

    return chosen.sort_values("p").set_index("p")


def make_figure(summary: pd.DataFrame, level: float = 0.8) -> plt.Figure:
    """Create the coverage-and-width figure from the study summary."""
    rows = _select(summary, level)
    supported = rows.loc[rows["supported"]]
    p_values = rows.index.tolist()

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
        }
    )

    fig, (ax_coverage, ax_width) = plt.subplots(
        1,
        2,
        figsize=(10, 4.4),
        facecolor=SURFACE,
        constrained_layout=True,
    )

    for ax in (ax_coverage, ax_width):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, length=0)
        ax.set_xticks(p_values)
        ax.set_xlabel("Number of training panels", color=INK_SECONDARY)

    # Left panel: how often a new household's yield falls inside each range.
    ax_coverage.axhline(
        level,
        color=INK_SECONDARY,
        linewidth=1,
        linestyle="--",
    )
    ax_coverage.text(
        p_values[0],
        level + 0.025,
        f"Target: {level:.0%}",
        color=INK_SECONDARY,
        fontsize=8.5,
    )

    ax_coverage.errorbar(
        supported.index,
        supported["coverage"],
        yerr=supported["coverage_se"],
        color=M1_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        capsize=3,
        label="Model interval",
    )

    history = rows.dropna(
        subset=["history_coverage", "history_coverage_se"]
    )
    ax_coverage.errorbar(
        history.index,
        history["history_coverage"],
        yerr=history["history_coverage_se"],
        color=HISTORY_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        capsize=3,
        label="Usual range of past yields",
    )

    # Coverage is a proportion, so starting at zero gives an honest visual
    # comparison instead of exaggerating small differences.
    ax_coverage.set_ylim(0, 1.05)
    ax_coverage.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_coverage.set_ylabel(
        "Share of households inside the range",
        color=INK_SECONDARY,
    )
    ax_coverage.set_title(
        "How often new yields fall inside each range",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    ax_coverage.legend(
        frameon=False,
        labelcolor=INK,
        loc="lower right",
    )

    # Right panel: the average width of each range. A logarithmic scale keeps
    # both the widest and narrowest intervals readable in the same panel.
    ax_width.plot(
        supported.index,
        supported["width"],
        color=M1_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="Model interval",
    )

    reference = rows.dropna(subset=["ref_width"])
    ax_width.plot(
        reference.index,
        reference["ref_width"],
        color=HISTORY_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="Usual range of past yields",
    )

    ax_width.set_yscale("log")
    width_ticks = [1, 2, 4, 8]
    ax_width.set_yticks(width_ticks)
    ax_width.set_yticklabels([str(tick) for tick in width_ticks])
    ax_width.minorticks_off()
    ax_width.set_ylabel("Range width (t/ha)", color=INK_SECONDARY)
    ax_width.set_title(
        "How wide each range is",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    ax_width.text(
        0,
        -0.2,
        "The vertical axis uses a logarithmic scale.",
        transform=ax_width.transAxes,
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    ax_width.legend(
        frameon=False,
        labelcolor=INK,
        loc="upper right",
    )

    unsupported = sorted(set(p_values) - set(supported.index))
    if unsupported:
        panel_text = ", ".join(str(value) for value in unsupported)
        ax_coverage.text(
            0,
            -0.2,
            f"A {level:.0%} interval cannot be calculated with "
            f"{panel_text} training panels.",
            transform=ax_coverage.transAxes,
            color=INK_SECONDARY,
            fontsize=8.5,
        )

    return fig


def main() -> None:
    """Read the study summary and save the finished figure."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--results",
        default="experiments/results",
        help="folder containing interval_summary.csv",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.8,
        help="interval level to plot",
    )
    args = parser.parse_args()

    folder = Path(args.results)
    summary_path = folder / "interval_summary.csv"

    if not summary_path.is_file():
        raise FileNotFoundError(f"could not find {summary_path}")

    summary = pd.read_csv(summary_path)
    fig = make_figure(summary, level=args.level)

    output_path = folder / "intervals.png"
    fig.savefig(
        output_path,
        dpi=200,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
