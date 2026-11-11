"""Export IJDC artwork from authenticated, unchanged experiment summaries.

Only paper/ijdc/figures is written. The original manuscript assets and their
manifests remain intact. PDF/EPS preserve vectors and embed TrueType fonts;
600-dpi PNG previews are also supplied at the intended 174-mm print width.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/ijdc/figures"
WIDTH = 174 / 25.4
REGIMES = ("id_clean", "id_corrupt", "ood_parameter_corrupt")
REGIME_LABELS = ("ID clean", "ID corrupt", "OOD + corrupt")
METHODS = ("rssm", "simnorm", "transformer", "pgnn", "pgdmm", "cfrwm")
LABELS = {"rssm": "RSSM-style", "simnorm": "SimNorm-style",
          "transformer": "Transformer-style", "pgnn": "PGNN-style",
          "pgdmm": "PG-DMM-style", "cfrwm": "CFR-WM"}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#666666")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticated_sources() -> dict[str, str]:
    """Verify all existing evidence hashes before reading plot inputs."""
    inputs = {}
    for name in ("manuscript_assets_manifest.json", "external_assets_manifest.json"):
        path = ROOT / "paper" / name
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise ValueError(f"incomplete evidence manifest: {name}")
        for group in ("generator", "authenticated_sources", "assets"):
            entries = manifest[group]
            if not entries:
                raise ValueError(f"empty evidence group: {name}/{group}")
            for relative, digest in entries.items():
                source = (ROOT / relative).resolve()
                if not source.is_relative_to(ROOT) or sha(source) != digest:
                    raise ValueError(f"changed evidence: {relative}")
                inputs[relative] = digest
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    return inputs


def export(fig: plt.Figure, number: int) -> None:
    for extension in ("pdf", "eps", "png"):
        options = {"metadata": {"Creator": "Matplotlib", "CreationDate": None}}
        if extension == "eps":
            options = {"metadata": {"Creator": "Matplotlib"}}
        elif extension == "png":
            options = {"metadata": {"Software": "Matplotlib"}}
        fig.savefig(OUT / f"Fig{number}.{extension}", dpi=600,
                    facecolor="white", **options)
    plt.close(fig)


def box(ax, xy, size, label, color, fontsize=8.5):
    x, y = xy
    w, h = size
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                               facecolor=color, edgecolor="#444444", linewidth=.7))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                linewidth=.8, color="#333333",
                                linestyle="--" if dashed else "-"))


def method_figure() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.25))
    fig.subplots_adjust(left=.02, right=.99, bottom=.03, top=.96, hspace=.28)
    for ax in axes:
        ax.set(xlim=(-.1, 10), ylim=(-.35, 3.0))
        ax.axis("off")
    ax = axes[0]
    ax.text(0, 2.85, "(a) Observed context", fontsize=10, weight="bold")
    box(ax, (0, .8), (2.15, 1.2), "8 context states\nvalues, masks, ages\nand past controls", "#EEEEEE")
    box(ax, (3, 1.45), (3.1, 1.0), "Nominal prediction + update\n" + r"$K_k=0.5m_k$" + "\n" + r"$\hat{x}_k=\Pi[\bar{x}_k+K_k(\tilde{y}_k-\bar{x}_k)]$", "#DFF0E3")
    box(ax, (3, -.05), (3.1, .9), "GRU context encoder", "#EEEEEE")
    box(ax, (7.2, 1.45), (2.4, 1.0), "Forecast origin\n" + r"$\hat{x}_t$", "#DFF0E3")
    box(ax, (7.2, -.05), (2.4, .9), "Recurrent context\n" + r"$b_t$", "#EEEEEE")
    arrow(ax, (2.2, 1.7), (2.95, 1.95))
    arrow(ax, (2.2, 1.05), (2.95, .4))
    arrow(ax, (6.15, 1.95), (7.15, 1.95))
    arrow(ax, (6.15, .4), (7.15, .4))
    ax = axes[1]
    ax.text(0, 2.85, "(b) Free rollout with known future controls", fontsize=10, weight="bold")
    box(ax, (0, .75), (2.15, 1.15), "Predicted state\nrecurrent context\ncontrol", "#EEEEEE")
    box(ax, (3, 1.5), (2.65, .9), "Nominal dynamics\n" + r"$F_0(\hat{x},u)$", "#DCEBFA")
    box(ax, (3, -.1), (2.65, 1.05), "Neural residual\nand variance\n" + r"$0.25\tanh r_\theta,\;\sigma^2_\theta$", "#FBE4D5")
    box(ax, (7.0, .9), (2.6, 1.1), "Predictive distribution\n" + r"$\mu=\Pi[F_0+0.25\tanh r_\theta]$" + "\n" + r"$\mathcal{N}(\mu,\mathrm{diag}\,\sigma^2)$", "#FBE4D5", 8.1)
    arrow(ax, (2.2, 1.65), (2.95, 1.95))
    arrow(ax, (2.2, .95), (2.95, .4))
    arrow(ax, (5.7, 1.95), (6.95, 1.65))
    arrow(ax, (5.7, .4), (6.95, 1.2))
    ax.plot([8.3, 8.3, 1.05], [.85, -.28, -.28], color="#333333", lw=.8, ls="--")
    arrow(ax, (1.05, -.28), (1.05, .7), dashed=True)
    ax.text(6.0, -.1, "Mean feedback + GRU update", ha="left", fontsize=8)
    export(fig, 1)


def overview(inference, summary, trajectories) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.4), layout="constrained")
    main = summary.set_index("method").loc[list(METHODS)]
    ax = axes[0, 0]
    ax.bar(np.arange(6), main.nrmse_h25_mean, yerr=main.nrmse_h25_sd,
           capsize=2, color=COLORS[0], error_kw={"elinewidth": .8})
    ax.set_xticks(np.arange(6), [LABELS[m] for m in METHODS], rotation=35, ha="right")
    ax.set_ylabel("NRMSE-25")
    ax.set_title("(a) Transition accuracy", loc="left")
    cells = trajectories.groupby(["method", "seed", "regime"])["nrmse_h25"].mean()
    paired = []
    for regime in REGIMES:
        baseline = cells.xs(("pgnn", regime), level=("method", "regime"))
        proposed = cells.xs(("cfrwm", regime), level=("method", "regime"))
        effects = 100 * (baseline - proposed) / baseline
        paired.append((effects.mean(), effects.std(ddof=1)))
    ax = axes[0, 1]
    values = np.asarray(paired)
    ax.bar(np.arange(3), values[:, 0], yerr=values[:, 1], capsize=2,
           color=[COLORS[2], "#E69F00", COLORS[1]], error_kw={"elinewidth": .8})
    ax.axhline(5, color="#333333", ls="--", lw=.8, label="5% margin")
    ax.axhline(0, color="#777777", lw=.6)
    ax.set_xticks(np.arange(3), REGIME_LABELS, rotation=25, ha="right")
    ax.set_ylabel("Improvement over PGNN-style (%)")
    ax.set_title("(b) Paired regime effects", loc="left")
    ax.legend(frameon=False)
    gain = summary.set_index("method").loc[["cfr_gain0", "cfr_gain025", "cfrwm", "cfr_gain075", "cfr_gain1"]]
    ax = axes[1, 0]
    ax.errorbar([0, .25, .5, .75, 1], gain.nrmse_h25_mean, yerr=gain.nrmse_h25_sd,
                fmt="o-", capsize=2, color=COLORS[2])
    ax.axvline(.5, color="#333333", ls="--", lw=.8)
    ax.set(xlabel="Fixed observed-channel gain", ylabel="NRMSE-25")
    ax.set_title("(c) Gain sensitivity", loc="left")
    contrasts = inference["comparator_contrasts"]
    points = np.asarray([c["relative_improvement_percent"] for c in contrasts])
    intervals = np.asarray([c["relative_improvement_ci_95"] for c in contrasts])
    ax = axes[1, 1]
    ax.errorbar(points, np.arange(len(contrasts)),
                xerr=np.maximum(0, np.vstack((points-intervals[:, 0], intervals[:, 1]-points))),
                fmt="o", color=COLORS[0], capsize=2)
    ax.axvline(0, color="#777777", lw=.6)
    ax.axvline(5, color="#333333", ls="--", lw=.8)
    ax.set_yticks(np.arange(len(contrasts)), [LABELS[c["baseline"]] for c in contrasts])
    ax.set_xlabel("CFR-WM relative improvement (%)")
    ax.set_title("(d) Paired bootstrap intervals", loc="left")
    export(fig, 2)


def heatmap(cell) -> None:
    systems = list(dict.fromkeys(cell.system))
    values = cell.pivot(index="system", columns="regime", values="relative_change_cfr_minus_pgnn_percent").loc[systems, list(REGIMES)].to_numpy()
    bound = max(5, float(np.max(np.abs(values))))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0), layout="constrained")
    # pcolormesh keeps the heatmap cells as vectors in both PDF and EPS.
    mesh = ax.pcolormesh(np.arange(4), np.arange(6), values, cmap="RdBu_r",
                         vmin=-bound, vmax=bound, shading="flat", edgecolor="white", linewidth=.6)
    for row in range(5):
        for col in range(3):
            rgba = mesh.cmap(mesh.norm(values[row, col]))
            lightness = np.dot(rgba[:3], [.299, .587, .114])
            ax.text(col+.5, row+.5, f"{values[row, col]:+.1f}%", ha="center", va="center",
                    color="black" if lightness > .55 else "white", fontsize=10)
    ax.set_xticks(np.arange(3)+.5, REGIME_LABELS)
    ax.set_yticks(np.arange(5)+.5, [s.replace("_", " ").title() for s in systems])
    ax.invert_yaxis()
    bar = fig.colorbar(mesh, ax=ax, label="CFR-WM minus PGNN-style (%)")
    bar.solids.set_rasterized(False)
    export(fig, 3)


def flight(summary) -> None:
    methods = ("cfr", "unfiltered", "recurrent", "nominal")
    regimes = ("recorded", "noise", "missing", "combined")
    labels = ("CFR-WM", "Unfiltered residual", "Recurrent", "Nominal prior")
    panels = ((25, "nrmse", "(a) 25-step forecast", "Normalized RMSE"),
              (50, "nrmse", "(b) 50-step forecast", "Normalized RMSE"),
              (50, "position_rmse_m", "(c) Position, 50 steps", "Position RMSE (m)"),
              (50, "velocity_rmse_m_s", "(d) Velocity, 50 steps", "Velocity RMSE (m/s)"))
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.2), layout="constrained")
    for ax, (horizon, metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        for i, method in enumerate(methods):
            frame = summary[(summary.method == method) & (summary.horizon == horizon)].set_index("regime").reindex(regimes)
            ax.errorbar(np.arange(4)+(i-1.5)*.13, frame[f"{metric}_mean"],
                        yerr=frame[f"{metric}_sd"], fmt=("o", "s", "^", "D")[i],
                        linestyle="none", color=COLORS[i], label=labels[i], capsize=2, markersize=4)
        ax.set_xticks(np.arange(4), ["Recorded", "Noise", "Missing", "Combined"], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#DDDDDD", lw=.5)
    axes[0, 0].legend(frameon=True, framealpha=1, fontsize=8)
    export(fig, 4)


def main() -> None:
    inputs = authenticated_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "lines.linewidth": 1, "lines.markersize": 4,
                         "xtick.major.width": .6, "ytick.major.width": .6,
                         "mathtext.fontset": "dejavusans"})
    base = ROOT / "results/confirmatory_analysis"
    method_figure()
    overview(json.loads((base / "confirmatory_inference.json").read_text()),
             pd.read_csv(base / "confirmatory_summary.csv"), pd.read_csv(base / "trajectory_metrics.csv"))
    heatmap(pd.read_csv(base / "cell_contrasts.csv"))
    flight(pd.read_csv(ROOT / "results/external_nanodrone/summary.csv"))
    artifacts = {p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.iterdir()
                 if p.suffix in (".pdf", ".eps", ".png") and p.stem in {f"Fig{i}" for i in range(1, 5)}}
    manifest = {"complete": True, "generator": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
                "authenticated_inputs": inputs, "artifacts": artifacts, "width_mm": 174,
                "font_points": {"minimum": 8, "body": 9, "panel_label": 10},
                "minimum_line_width_pt": .5, "raster_dpi": 600,
                "numeric_changes": "none; unchanged aggregations and source values",
                "formats": "vector PDF and EPS with embedded fonts; RGB PNG previews"}
    (OUT / "artwork_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Exported {len(artifacts)} artwork files at {OUT}")


if __name__ == "__main__":
    main()
