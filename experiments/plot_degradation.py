"""Plot the degradation study results.

Example
-------
python experiments/plot_degradation.py

Reads ``degradation_results.csv`` from the results folder and writes
``degradation.png`` next to it. Both panels use the runs with no household
cap, so the only thing that changes along the x-axis is the number of
training panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"
M0_COLOR = "#2a78d6"
M1_COLOR = "#eb6834"
PANEL_GRAY = "#b9b8b2"


def _panel_means(results: pd.DataFrame) -> pd.DataFrame:
    """Average the draws, keeping runs with no household cap."""
    uncapped = results[np.isinf(results["h"])]
    return uncapped.groupby(
        ["model", "held_out", "p"],
        as_index=False,
    )["mae"].mean()


def _leave_room_for_legend(ax: plt.Axes, values: pd.Series) -> None:
    """Extend the y-axis upward so the legend never covers a line."""
    low, high = float(values.min()), float(values.max())
    span = high - low
    ax.set_ylim(low - 0.08 * span, high + 0.45 * span)


def make_figure(results: pd.DataFrame) -> plt.Figure:
    """Build the two-panel degradation figure from the raw study results."""
    results = results.assign(h=pd.to_numeric(results["h"]))
    per_panel = _panel_means(results)

    mean_mae = per_panel.groupby(["model", "p"])["mae"].mean().unstack("model")
    wide = per_panel.pivot_table(
        index=["held_out", "p"],
        columns="model",
        values="mae",
    ).reset_index()
    wide["skill"] = 100 * (1 - wide["m1"] / wide["m0"])
    skill_mean = wide.groupby("p")["skill"].mean()
    n_panels = wide["held_out"].nunique()
    p_values = sorted(wide["p"].unique())

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, (ax_mae, ax_skill) = plt.subplots(
        1,
        2,
        figsize=(10, 4.2),
        facecolor=SURFACE,
        constrained_layout=True,
    )

    for ax in (ax_mae, ax_skill):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, length=0)
        ax.set_xticks(p_values)
        ax.set_xlabel("Training panels", color=INK_SECONDARY)

    # Left: average error for each model.
    for name, color, label in (
        ("m0", M0_COLOR, "M0 season-district table"),
        ("m1", M1_COLOR, "M1 ridge"),
    ):
        ax_mae.plot(
            p_values,
            mean_mae[name].reindex(p_values),
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
    _leave_room_for_legend(ax_mae, mean_mae.stack())
    ax_mae.set_ylabel("Mean absolute error (t/ha)", color=INK_SECONDARY)
    ax_mae.set_title(
        "Error on an unseen panel",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    ax_mae.text(
        0,
        -0.2,
        "Axis does not start at zero.",
        transform=ax_mae.transAxes,
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    ax_mae.legend(frameon=False, labelcolor=INK, loc="upper right")

    # Right: M1 skill over M0 for each held-out panel.
    for _, group in wide.groupby("held_out"):
        ax_skill.plot(
            group["p"],
            group["skill"],
            color=PANEL_GRAY,
            linewidth=1,
            alpha=0.9,
        )
    ax_skill.plot(
        [],
        [],
        color=PANEL_GRAY,
        linewidth=1,
        label=f"Each held-out panel ({n_panels})",
    )
    ax_skill.plot(
        p_values,
        skill_mean.reindex(p_values),
        color=M1_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="Average over panels",
    )
    ax_skill.axhline(0, color=INK_SECONDARY, linewidth=1)
    _leave_room_for_legend(ax_skill, pd.concat([wide["skill"], pd.Series([0.0])]))
    ax_skill.set_ylabel("M1 error reduction vs M0 (%)", color=INK_SECONDARY)
    ax_skill.set_title(
        "M1 gain over M0",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    ax_skill.legend(frameon=False, labelcolor=INK, loc="upper left")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="experiments/results")
    args = parser.parse_args()

    folder = Path(args.results)
    results = pd.read_csv(folder / "degradation_results.csv")
    fig = make_figure(results)
    out = folder / "degradation.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
