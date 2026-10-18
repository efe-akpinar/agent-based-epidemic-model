"""Figures for a simulation run."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from .analysis import kaplan_meier, survival_frame  # noqa: E402
from .model import IMMUNITY_CLASSES, SimulationResult  # noqa: E402
from .symptoms import SYMPTOMS  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SYMPTOM_ORDER = list(SYMPTOMS)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
})

compact = FuncFormatter(lambda v, _: f"{v / 1000:g}K" if abs(v) >= 1000 else f"{v:g}")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _phase_markers(ax, result: SimulationResult) -> None:
    for day, label in ((result.epidemic_day, "epidemic day"), (result.normalization_day, "normalization")):
        if day is not None:
            ax.axvline(day, color=INK_MUTED, linewidth=1)
            near_end = day > 0.75 * result.scenario.n_days
            ax.annotate(f"{label} ({day})", (day, 1), xycoords=("data", "axes fraction"),
                        xytext=(-4 if near_end else 4, -4), textcoords="offset points", va="top",
                        ha="right" if near_end else "left", fontsize=8, color=INK_SECONDARY)


def _suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.06, y=1.04, ha="left", fontsize=13, fontweight="bold")


def _end_labels(ax, x, series: Sequence[tuple[str, Sequence[float]]], min_gap: float) -> None:
    """Direct labels at line ends; labels that would collide are left to the legend."""
    placed: list[float] = []
    for label, values in sorted(series, key=lambda s: -s[1][-1]):
        y = values[-1]
        if all(abs(y - p) >= min_gap for p in placed):
            ax.annotate(f"{label} {y:,.0f}", (x[-1], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_SECONDARY)
            placed.append(y)


def plot_sird(result: SimulationResult, path: Path) -> Path:
    d = result.daily
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.5})
    series = [("Susceptible", d["S"]), ("Infected", d["I"]), ("Recovered", d["R"]), ("Dead", d["D"])]
    for (label, values), color in zip(series, SERIES):
        left.plot(d["day"], values, color=color, label=label)
    _end_labels(left, d["day"].values, [(k, v.values) for k, v in series], min_gap=0.06 * result.config.n_agents)
    _phase_markers(left, result)
    left.set_title("Agents by state")
    left.set(xlabel="Day", ylabel="Agents")
    left.yaxis.set_major_formatter(compact)
    left.set_ylim(0, result.config.n_agents * 1.04)
    left.margins(x=0)
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)

    right.plot(d["day"], d["mean_beta"], color=SERIES[0], label="Mean β (transmission)")
    right.plot(d["day"], d["mean_gamma"], color=SERIES[1], label="Mean γ (recovery)")
    _phase_markers(right, result)
    right.set_title("Mean β and γ of infected agents")
    right.set(xlabel="Day", ylabel="Rate per day")
    right.set_ylim(bottom=0)
    right.margins(x=0)
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _suptitle(fig, result.scenario.name)
    return _save(fig, path)


def plot_r0(result: SimulationResult, path: Path, days: int = 30) -> Path:
    d = result.daily[result.daily["day"] < days]
    peak = d.loc[d["r0"].idxmax()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["day"], d["r0"], color=SERIES[0])
    ax.axhline(1, color=INK_MUTED, linewidth=1)
    ax.annotate("R0 = 1", (d["day"].iloc[-1], 1), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK_SECONDARY)
    ax.plot(peak["day"], peak["r0"], "o", markersize=8, color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"peak {peak['r0']:.2f} (day {int(peak['day'])})", (peak["day"], peak["r0"]),
                xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK_SECONDARY)
    ax.set_title(f"{result.scenario.name}: R0 = mean β / mean γ")
    ax.set(xlabel="Day", ylabel="R0")
    ax.set_ylim(0, max(2.0, peak["r0"] * 1.15))
    ax.margins(x=0.01)
    return _save(fig, path)


def plot_symptoms(result: SimulationResult, path: Path) -> Path:
    counts = result.symptoms
    columns = [s for s in SYMPTOM_ORDER if s in counts.columns]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    days = np.arange(len(counts))
    if len(days) <= 40:
        bottom = np.zeros(len(days))
        for symptom, color in zip(columns, SERIES):
            ax.bar(days, counts[symptom], bottom=bottom, width=0.7, color=color, label=symptom,
                   edgecolor=SURFACE, linewidth=1)
            bottom += counts[symptom].values
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.stackplot(days, [counts[s] for s in columns], colors=SERIES[:len(columns)], labels=columns,
                     edgecolor=SURFACE, linewidth=1)
    ax.set_title(f"{result.scenario.name}: active symptoms across infected agents")
    ax.set(xlabel="Day", ylabel="Symptom count")
    ax.yaxis.set_major_formatter(compact)
    ax.margins(x=0.01)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Symptom", title_fontsize=9)
    return _save(fig, path)


def plot_hospital(result: SimulationResult, path: Path) -> Path:
    d = result.daily
    units = [("General ward", d["hospital_general"]), ("ICU", d["hospital_icu"]),
             ("Pandemic hospital", d["hospital_pandemic"])]
    fig, (load, queue) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.3})
    load.stackplot(d["day"], [v for _, v in units], colors=SERIES[:3], labels=[k for k, _ in units],
                   edgecolor=SURFACE, linewidth=1)
    load.step(d["day"], d["hospital_capacity"], where="post", color=INK_SECONDARY, linewidth=1.2, label="Bed capacity")
    _phase_markers(load, result)
    load.set_title("Hospital occupancy by unit")
    load.set(xlabel="Day", ylabel="Patients")
    load.set_ylim(0, d["hospital_capacity"].max() * 1.15)
    load.margins(x=0)
    load.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)

    queue.plot(d["day"], d["turned_away"], color=SERIES[0])
    _phase_markers(queue, result)
    queue.set_title("Severe cases turned away (treated at home)")
    queue.set(xlabel="Day", ylabel="Agents")
    queue.yaxis.set_major_formatter(compact)
    queue.set_ylim(bottom=0)
    queue.margins(x=0)
    _suptitle(fig, result.scenario.name)
    return _save(fig, path)


def plot_recovery_times(result: SimulationResult, path: Path) -> Path:
    a = result.agents[result.agents["outcome"] == "recovered"]
    days = a["outcome_day"] - a["infection_day"]
    groups = [(c, days[a["immunity"] == c].values) for c in IMMUNITY_CLASSES]
    groups = [(c, v) for c, v in groups if len(v)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    parts = ax.violinplot([v for _, v in groups], showextrema=False, widths=0.7)
    for body in parts["bodies"]:
        body.set_facecolor(SERIES[0])
        body.set_edgecolor(SERIES[0])
        body.set_alpha(0.25)
    for i, (_, v) in enumerate(groups, start=1):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, v.min(), v.max(), color=SERIES[0], linewidth=1)
        ax.vlines(i, q1, q3, color=SERIES[0], linewidth=5)
        ax.plot(i, med, "o", markersize=8, color=SURFACE, markeredgecolor=SERIES[0], markeredgewidth=2)
        ax.annotate(f"median {med:g}, max {v.max():g}", (i, v.max()), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(range(1, len(groups) + 1), [f"{c.capitalize()} immunity\n(n = {len(v):,})" for c, v in groups])
    ax.set_title(f"{result.scenario.name}: days to recovery")
    ax.set_ylabel("Days from infection to recovery")
    ax.set_ylim(0, max(v.max() for _, v in groups) * 1.12)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_survival(result: SimulationResult, path: Path) -> Path:
    frame = survival_frame(result)
    panels = [
        ("All infected agents", [("All infected", frame)]),
        ("Care setting", [(k.capitalize(), g) for k, g in frame.groupby("care")]),
        ("Symptom severity", [(k.capitalize(), g) for k, g in frame.groupby("severity")]),
        ("Hospital unit", [(k if k == "ICU" else k.capitalize(), g) for k, g in frame.groupby("unit")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.2})
    for ax, (title, groups) in zip(axes.flat, panels):
        for (label, group), color in zip(groups, SERIES):
            km = kaplan_meier(group["duration"], group["event"])
            ax.step(km["time"], km["survival"], where="post", color=color, label=f"{label} (n = {len(group):,})")
            ax.fill_between(km["time"], km["lower"], km["upper"], step="post", color=color, alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.04)
        ax.legend(loc="best")
    for ax in axes[1]:
        ax.set_xlabel("Days since infection")
    for ax in axes[:, 0]:
        ax.set_ylabel("Survival probability")
    fig.suptitle(f"{result.scenario.name}: Kaplan–Meier survival", x=0.06, ha="left", fontsize=13,
                 fontweight="bold")
    return _save(fig, path)


def plot_outcomes(results: Sequence[SimulationResult], path: Path) -> Path:
    """Final state of the population in each scenario, as shares of all agents."""
    # Same entity colours as the SIRD charts: S blue, I orange, R aqua, D yellow.
    states = [("Never infected", "S", SERIES[0]), ("Recovered", "R", SERIES[2]),
              ("Still infected", "I", SERIES[1]), ("Dead", "D", SERIES[3])]
    fig, ax = plt.subplots(figsize=(10, 1.1 + 0.55 * len(results)))
    names = [f"{r.scenario.name}\n({r.scenario.n_days} days)" for r in results][::-1]
    left = np.zeros(len(results))
    for label, key, color in states:
        shares = np.array([r.daily[key].iloc[-1] / r.config.n_agents for r in results])[::-1]
        ax.barh(names, shares, left=left, height=0.42, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for y, (x0, w) in enumerate(zip(left, shares)):
            if w >= 0.07:
                ax.text(x0 + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if color in (SERIES[0], SERIES[1]) else INK)
        left += shares
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Population outcome at the end of each scenario")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    return _save(fig, path)
