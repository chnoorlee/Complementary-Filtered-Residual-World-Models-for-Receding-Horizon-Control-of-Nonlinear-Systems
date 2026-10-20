"""Render the frozen CFR-WM causal data-flow diagram."""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(axis, x, y, width, height, text, face, edge, size=9):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.02",
        linewidth=1.4, facecolor=face, edgecolor=edge,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size)


def arrow(axis, start, end, color="#444444", style="-"):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.3,
        linestyle=style, color=color, connectionstyle="arc3,rad=0.0",
    ))


def main():
    figure, axis = plt.subplots(figsize=(12.2, 5.4), constrained_layout=True)
    axis.set_xlim(0, 12.4)
    axis.set_ylim(0, 6.0)
    axis.axis("off")
    gray, blue, orange, green = "#EEF1F4", "#DCEBFA", "#FBE4D5", "#DFF0E3"

    box(axis, 0.2, 3.75, 2.15, 1.15,
        "8-step corrupted context\nvalues, masks, elapsed time\npast actions", gray, "#66727E")
    box(axis, 2.75, 4.0, 1.55, 0.9, "GRU context\n$\\mathbf{b}_t$", gray, "#66727E")
    box(axis, 2.75, 1.15, 1.75, 1.0,
        "Nominal RK4\nprediction $\\bar{\\mathbf{x}}_k$", blue, "#3274A1")
    box(axis, 5.0, 1.05, 2.1, 1.2,
        "Mask-exact update\n$K_k=0.5m_k$\n$\\hat{x}_k=\\Pi[\\bar{x}_k+K_k(y_k-\\bar{x}_k)]$",
        green, "#3A8754", 8.2)
    box(axis, 7.65, 1.2, 1.75, 0.9, "Filtered forecast\norigin $\\hat{\\mathbf{x}}_t$", green, "#3A8754")
    box(axis, 7.55, 4.0, 1.95, 0.95,
        "Nominal RK4 step\n$F_0(\\hat{x},u)$", blue, "#3274A1")
    box(axis, 9.9, 4.0, 2.0, 0.95,
        "Bounded residual + variance\n$0.25\\tanh r_\\theta,\\;\\sigma^2_\\theta$",
        orange, "#C86B31", 8.4)
    box(axis, 9.9, 2.45, 2.0, 0.85, "Predictive state\n$\\mathcal{N}(\\mu_{t+h},\\mathrm{diag}\\,\\sigma^2_{t+h})$",
        orange, "#C86B31", 8.2)

    arrow(axis, (2.35, 4.35), (2.75, 4.45))
    arrow(axis, (2.35, 4.0), (3.55, 2.15), "#3274A1")
    arrow(axis, (4.5, 1.65), (5.0, 1.65), "#3A8754")
    arrow(axis, (7.1, 1.65), (7.65, 1.65), "#3A8754")
    arrow(axis, (9.4, 1.65), (8.5, 4.0), "#3A8754")
    arrow(axis, (4.3, 4.45), (9.9, 4.55), "#66727E")
    arrow(axis, (9.5, 4.5), (9.9, 4.5), "#3274A1")
    arrow(axis, (10.9, 4.0), (10.9, 3.3), "#C86B31")
    arrow(axis, (9.9, 2.85), (8.55, 4.0), "#666666", "--")

    axis.text(0.25, 0.35,
              "Context filtering is causal and fixed before training; the 25-step rollout receives future controls but no future observations.",
              fontsize=9.2, weight="bold")
    axis.text(4.85, 2.45, "repeat within observed context", fontsize=8, color="#3A8754")
    axis.text(7.6, 3.25, "free-rollout feedback", fontsize=8, color="#555555", rotation=20)

    destination = Path("paper/mypaper/figures/figure1_method.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, facecolor="white")
    plt.close(figure)
    archive = Path("figures/figure1_method.png")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination, archive)
    print(destination)


if __name__ == "__main__":
    main()
