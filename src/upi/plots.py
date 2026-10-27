"""Consistent, report-ready charts (static PNGs for the README and reports).

Palette: a validated colour-blind-safe categorical order; sequential blues for
magnitude. One y-axis per chart -- two measures get two panels, never twin axes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:   # headless for scripts; inline charts in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from upi import config  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#b5b3ad"
TEXT, TEXT_2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e9e8e4",
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "legend.frameon": False,
    "font.size": 10,
    "lines.linewidth": 2,
})


def save(fig: plt.Figure, name: str, fig_dir: Path = config.FIG_DIR) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _label_line_ends(ax, x, series: dict[str, np.ndarray], fmt="{:.0%}"):
    for i, (label, y) in enumerate(series.items()):
        ax.annotate(f"{label}  {fmt.format(y[-1])}", (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXT_2, fontsize=9)


# ------------------------------------------------------------------ growth
def growth_overview(kpis: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(kpis))
    labels = kpis["month"].str[2:]
    panels = [
        ("New users per month", "new_users", "{:,.0f}"),
        ("Monthly active users", "mau", "{:,.0f}"),
        ("Monthly revenue (Rs)", "revenue", "{:,.0f}"),
    ]
    for ax, (title, col, fmt) in zip(axes, panels):
        if col == "new_users":
            ax.bar(x, kpis[col], color=SERIES[0], width=0.75)
            neighbours = (kpis[col].shift(1) + kpis[col].shift(-1)) / 2
            for i in (kpis[col] / neighbours).nlargest(2).index:   # spikes vs adjacent months
                ax.annotate("campaign", (x[i], kpis[col][i]), xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8, color=TEXT_2)
        else:
            ax.plot(x, kpis[col], color=SERIES[0], marker="o", markersize=3)
            ax.annotate(fmt.format(kpis[col].iloc[-1]), (x[-1], kpis[col].iloc[-1]), xytext=(-4, 8),
                        textcoords="offset points", ha="right", fontsize=9, color=TEXT_2)
        ax.set_title(title)
        ax.set_xticks(x[::3], labels[::3])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_ylim(bottom=0)
    return fig


def cohort_heatmap(matrix: pd.DataFrame, max_months: int = 12) -> plt.Figure:
    m = matrix.iloc[:, : max_months + 1]
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(m.to_numpy(dtype=float), cmap=BLUES, vmin=0, vmax=0.75, aspect="auto")
    for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.45 else TEXT)
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Signup cohort")
    ax.set_title("Cohort retention: share of each signup cohort transacting in month k")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, format=matplotlib.ticker.PercentFormatter(1.0))
    return fig


def retention_curves(curves: pd.DataFrame, title: str, labels: dict | None = None) -> plt.Figure:
    """``curves``: index = months since signup, one column per segment."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ends = {}
    for i, col in enumerate(curves.columns):
        name = (labels or {}).get(col, str(col))
        ax.plot(curves.index, curves[col], color=SERIES[i], marker="o", markersize=4, label=name)
        ends[name] = curves[col].dropna().to_numpy()
    _label_line_ends(ax, curves.index.to_numpy(), ends)
    ax.set_xlim(right=curves.index.max() + 3)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Users active")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return fig


def funnel(f: pd.DataFrame) -> plt.Figure:
    """``f``: columns step, users, pct_of_signups."""
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(f))[::-1]
    ax.barh(y, f["users"], color=SERIES[0], height=0.6)
    for yi, (_, row) in zip(y, f.iterrows()):
        ax.text(row["users"], yi, f"  {row['users']:,.0f}  ({row['pct_of_signups']:.0%})",
                va="center", fontsize=9, color=TEXT_2)
    ax.set_yticks(y, f["step"])
    ax.set_xlim(0, f["users"].max() * 1.3)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Onboarding funnel (users with 60+ days of history)")
    return fig


def bar_by_segment(df: pd.DataFrame, x: str, y: str, title: str, fmt: str = "Rs {:,.1f}") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = np.arange(len(df))
    ax.bar(pos, df[y], color=SERIES[0], width=0.6)
    for p, v in zip(pos, df[y]):
        ax.annotate(fmt.format(v), (p, v), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, color=TEXT_2)
    ax.set_xticks(pos, df[x].astype(str))
    ax.set_title(title)
    ax.set_ylim(0, df[y].max() * 1.18)
    return fig


def pareto(curve: pd.DataFrame, top_share: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["user_share"], curve["revenue_share"], color=SERIES[0])
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(0.2, color=MUTED, linewidth=1)
    ax.annotate(f"Top 20% of users = {top_share:.0%} of revenue", (0.2, top_share), xytext=(10, -14),
                textcoords="offset points", fontsize=9, color=TEXT)
    ax.scatter([0.2], [top_share], color=SERIES[0], s=36, zorder=3, edgecolor=SURFACE, linewidth=2)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of transacting users (highest revenue first)")
    ax.set_ylabel("Cumulative share of revenue")
    ax.set_title("Revenue concentration (Pareto curve)")
    return fig


# ------------------------------------------------------------------ A/B tests
def ab_forest(results: pd.DataFrame) -> plt.Figure:
    """``results``: experiment, metric, rel_lift, rel_ci_low, rel_ci_high, significant."""
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(results) + 1.5))
    y = np.arange(len(results))[::-1]
    for yi, (_, r) in zip(y, results.iterrows()):
        color = SERIES[0] if r["significant"] else MUTED
        ax.plot([r["rel_ci_low"], r["rel_ci_high"]], [yi, yi], color=color, linewidth=2)
        ax.scatter([r["rel_lift"]], [yi], color=color, s=40, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, results["experiment"] + "  ·  " + results["metric"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Relative lift vs control (95% CI) — blue = significant after Holm correction")
    ax.set_title("A/B test results")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return fig


# ------------------------------------------------------------------ fraud
def pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], base_rate: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, (precision, recall, ap)) in enumerate(curves.items()):
        ax.plot(recall, precision, color=SERIES[i], label=f"{name} (PR-AUC {ap:.3f})")
    ax.axhline(base_rate, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"random = fraud rate {base_rate:.1%}", (0.99, base_rate), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=TEXT_2)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on the held-out test period")
    ax.legend(loc="upper right")
    return fig


def threshold_cost(df: pd.DataFrame, best: float) -> plt.Figure:
    """``df``: threshold, total_cost, fraud_loss, friction_cost."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["threshold"], df["total_cost"], color=SERIES[0], label="Total cost")
    ax.plot(df["threshold"], df["fraud_loss"], color=SERIES[1], label="Missed-fraud loss")
    ax.plot(df["threshold"], df["friction_cost"], color=SERIES[2], label="False-positive friction")
    ax.axvline(best, color=TEXT_2, linewidth=1, linestyle="--")
    ax.annotate(f"chosen threshold {best:.2f}", (best, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=TEXT)
    ax.set_xlabel("Decision threshold (block if score ≥ threshold)")
    ax.set_ylabel("Cost on validation period (USD)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:,.0f}K"))
    ax.set_title("Choosing the threshold by business cost")
    ax.legend(loc="center right")
    return fig


def fraud_rate_by(df: pd.DataFrame, col: str, title: str) -> plt.Figure:
    g = df.groupby(col, observed=True)["isFraud"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = np.arange(len(g))
    ax.bar(pos, g["mean"], color=SERIES[0], width=0.6)
    for p, (rate, n) in zip(pos, g.itertuples(index=False)):
        ax.annotate(f"{rate:.1%}\n(n={n:,})", (p, rate), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_2)
    ax.set_xticks(pos, g.index.astype(str))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, g["mean"].max() * 1.3)
    ax.set_title(title)
    return fig


def feature_importance(imp: pd.Series, title: str) -> plt.Figure:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh(np.arange(len(imp)), imp.to_numpy(), color=SERIES[0], height=0.65)
    ax.set_yticks(np.arange(len(imp)), imp.index)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_title(title)
    return fig
