"""Figures. Every function returns the figure, or saves it and returns the path when ``path`` is given."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from .analysis import kaplan_meier, reproduction_by_cohort, survival_frame  # noqa: E402
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
# Entity colours shared by every chart that shows disease states.
STATES = [("Susceptible", "S", SERIES[0]), ("Infected", "I", SERIES[1]),
          ("Recovered", "R", SERIES[2]), ("Dead", "D", SERIES[3])]

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
percent = FuncFormatter(lambda v, _: f"{v:.0%}")


def _finish(fig, path: Path | str | None):
    if path is None:
        return fig
    path = Path(path)
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


def _suptitle(fig, text: str, y: float = 1.04) -> None:
    fig.suptitle(text, x=0.06, y=y, ha="left", fontsize=13, fontweight="bold")


def _end_labels(ax, x, series: Sequence[tuple[str, Sequence[float]]], min_gap: float) -> None:
    """Direct labels at line ends; labels that would collide are left to the legend."""
    placed: list[float] = []
    for label, values in sorted(series, key=lambda s: -s[1][-1]):
        y = values[-1]
        if all(abs(y - p) >= min_gap for p in placed):
            ax.annotate(f"{label} {y:,.0f}", (x[-1], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_SECONDARY)
            placed.append(y)


def _dot(ax, x, y, color, **kwargs) -> None:
    ax.plot(x, y, "o", markersize=7, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, **kwargs)


# --------------------------------------------------------------------------- single runs
def plot_sird(result: SimulationResult, path=None):
    d = result.daily
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.5})
    for label, key, color in STATES:
        left.plot(d["day"], d[key], color=color, label=label)
    _end_labels(left, d["day"].values, [(label, d[key].values) for label, key, _ in STATES],
                min_gap=0.06 * result.config.n_agents)
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
    return _finish(fig, path)


def plot_r0(result: SimulationResult, path=None, days: int = 30):
    d = result.daily[(result.daily["day"] < days) & result.daily["r0"].notna()]
    peak = d.loc[d["r0"].idxmax()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["day"], d["r0"], color=SERIES[0])
    ax.axhline(1, color=INK_MUTED, linewidth=1)
    ax.annotate("R0 = 1", (d["day"].iloc[-1], 1), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK_SECONDARY)
    _dot(ax, peak["day"], peak["r0"], SERIES[0])
    ax.annotate(f"peak {peak['r0']:.2f} (day {int(peak['day'])})", (peak["day"], peak["r0"]),
                xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK_SECONDARY)
    ax.set_title(f"{result.scenario.name}: R0 = mean β / mean γ")
    ax.set(xlabel="Day", ylabel="R0")
    ax.set_ylim(0, max(2.0, peak["r0"] * 1.15))
    ax.margins(x=0.01)
    return _finish(fig, path)


def plot_reproduction(result: SimulationResult, path=None, days: int = 20):
    """Measured case reproduction number (who infected whom) against the report's beta/gamma ratio."""
    cohorts = reproduction_by_cohort(result)
    cohorts = cohorts[(cohorts["infection_day"] < days) & (cohorts["cases"] >= 20)]
    d = result.daily[(result.daily["day"] < days) & result.daily["r0"].notna()]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(cohorts["infection_day"], cohorts["r_case"], color=SERIES[0],
            label="Measured: secondary infections per case, by day of infection")
    ax.plot(d["day"], d["r0"], color=SERIES[1], label="Report's R0 = mean β / mean γ")
    ax.axhline(1, color=INK_MUTED, linewidth=1)
    ax.set_title(f"{result.scenario.name}: reproduction number")
    ax.set(xlabel="Day", ylabel="Reproduction number")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    return _finish(fig, path)


def plot_symptoms(result: SimulationResult, path=None):
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
    return _finish(fig, path)


def plot_hospital(result: SimulationResult, path=None):
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
    load.set_ylim(0, max(1, d["hospital_capacity"].max()) * 1.15)
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
    return _finish(fig, path)


def plot_recovery_times(result: SimulationResult, path=None):
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
    return _finish(fig, path)


def plot_survival(result: SimulationResult, path=None):
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
        if groups:
            ax.legend(loc="best")
    for ax in axes[1]:
        ax.set_xlabel("Days since infection")
    for ax in axes[:, 0]:
        ax.set_ylabel("Survival probability")
    _suptitle(fig, f"{result.scenario.name}: Kaplan–Meier survival", y=0.98)
    return _finish(fig, path)


def plot_outcomes(results: Sequence[SimulationResult], path=None):
    """Final state of the population in each scenario, as shares of all agents."""
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
    ax.xaxis.set_major_formatter(percent)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Population outcome at the end of each scenario")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    return _finish(fig, path)


# --------------------------------------------------------------------------- ensembles
def plot_ensembles(ensembles, path=None):
    """Median and 90% interval of the SIRD curves across seeds, one panel per scenario."""
    fig, axes = plt.subplots(1, len(ensembles), figsize=(4.4 * len(ensembles), 4.2), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    for ax, ens in zip(axes, ensembles):
        bands = ens.bands([key for _, key, _ in STATES])
        for label, key, color in STATES:
            ax.fill_between(bands.index, bands[(key, 0.05)], bands[(key, 0.95)], color=color, alpha=0.2, linewidth=0)
            ax.plot(bands.index, bands[(key, 0.5)], color=color, label=label)
        ax.set_title(ens.scenario.name.replace(" infection", ""), fontsize=11)
        ax.set_xlabel("Day")
        ax.margins(x=0)
        ax.yaxis.set_major_formatter(compact)
    axes[0].set_ylabel("Agents")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5 * len(ensembles) + 0.06 * (len(ensembles) - 1), -0.16),
                   ncol=4)
    n_runs = len(ensembles[0].summaries)
    _suptitle(fig, f"Median and 90% interval across {n_runs} seeds per scenario", y=1.03)
    return _finish(fig, path)


def plot_progression(fixed, stochastic, path=None):
    """Severe-stage share and case fatality by immunity class: fixed vs stochastic progression."""
    def ratio(ens, numerator):
        return np.array([[s[numerator][c] / max(1, s["infected_by_immunity"][c]) for c in IMMUNITY_CLASSES]
                         for s in ens.summaries])

    panels = [("Infected agents reaching the severe stage", "severe_by_immunity"),
              ("Case fatality", "deaths_by_immunity")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.25})
    x = np.arange(len(IMMUNITY_CLASSES))
    for ax, (title, key) in zip(axes, panels):
        for offset, (label, ens), color in zip((-0.12, 0.12), (("Fixed timeline (report)", fixed),
                                                             ("Stochastic progression", stochastic)), SERIES):
            values = ratio(ens, key)
            low, med, high = np.quantile(values, [0.05, 0.5, 0.95], axis=0)
            ax.vlines(x + offset, low, high, color=color, linewidth=2)
            _dot(ax, x + offset, med, color, label=label)
            for xi, m in zip(x + offset, med):
                ax.annotate(f"{m:.0%}", (xi, m), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK_SECONDARY)
        ax.set_xticks(x, [f"{c.capitalize()} immunity" for c in IMMUNITY_CLASSES])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(percent)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, len(IMMUNITY_CLASSES) - 0.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=2)
    _suptitle(fig, f"{fixed.scenario.name}: disease course by immunity (median and 90% interval)", y=1.05)
    return _finish(fig, path)


# --------------------------------------------------------------------------- sensitivity
def plot_morris(indices: pd.DataFrame, path=None, order_by: str = "deaths"):
    outputs = list(dict.fromkeys(indices["output"]))
    ranking = indices[indices["output"] == order_by].sort_values("relative")
    order, labels = ranking["factor"].tolist(), ranking["factor_label"].tolist()
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, len(outputs), figsize=(4.2 * len(outputs) + 1.5, 0.36 * len(order) + 1.6),
                             sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, output in zip(axes, outputs):
        g = indices[indices["output"] == output].set_index("factor").loc[order]
        scale = g["mu_star"].max()
        ax.barh(y, g["relative"], height=0.42, color=SERIES[0])
        ax.errorbar(g["relative"], y, xerr=g["mu_star_conf"] / scale, fmt="none", ecolor=INK_SECONDARY,
                    elinewidth=1, capsize=0)
        ax.set_title(g["output_label"].iloc[0])
        ax.set_xlim(0, 1.45)
        ax.set_xlabel("μ* relative to the largest effect")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, labels)
    _suptitle(fig, "Morris sensitivity of the lower respiratory scenario to settings the report leaves open", y=1.02)
    return _finish(fig, path)


# --------------------------------------------------------------------------- calibration
def plot_calibration(calibration, sir: dict | None = None, path=None):
    from .calibration import observation_days, sir_curve

    obs = calibration.observed
    dates = obs["date"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"wspace": 0.22}, sharey=True)
    panels = [("in_bed", "Boys confined to bed (fitted)"), ("convalescent", "Boys convalescing (held out)")]
    for ax, (column, title) in zip(axes, panels):
        b = calibration.bands(column)
        ax.fill_between(dates, b[0.05], b[0.95], color=SERIES[0], alpha=0.18, linewidth=0)
        ax.plot(dates, b[0.5], color=SERIES[0], label="Agent-based model: median and 90% interval")
        if column == "in_bed" and sir is not None:
            curve = sir_curve(sir["beta"], sir["gamma"], sir["t0"], observation_days(obs))
            ax.plot(dates, curve, color=SERIES[1], label=f"Deterministic SIR fit (R0 = {sir['r0']:.1f})")
        _dot(ax, dates, obs[column], INK, label="Observed, 1978")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Boys")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               labelcolor=INK_SECONDARY)
    _suptitle(fig, "Boarding-school influenza, 1978: calibrated model against the observed outbreak", y=1.03)
    return _finish(fig, path)


def plot_posterior(calibration, path=None):
    from .calibration import PRIORS

    labels = {"beta_scale": "Transmission scale", "latent_days": "Latent days", "prodromal_days": "Prodromal days",
              "bed_days": "Days in bed", "bed_probability": "Infections confined to bed"}
    fig, axes = plt.subplots(1, len(PRIORS), figsize=(2.7 * len(PRIORS), 2.8), gridspec_kw={"wspace": 0.35})
    for ax, (name, prior) in zip(axes, PRIORS.items()):
        values = calibration.accepted[name]
        if isinstance(prior, tuple):
            ax.hist(values, bins=np.linspace(*prior, 13), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
            ax.set_xlim(*prior)
        else:
            counts = values.value_counts().reindex(prior, fill_value=0)
            ax.bar(counts.index, counts.values, width=0.5, color=SERIES[0])
            ax.set_xticks(prior)
        ax.set_title(labels[name], fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Accepted draws")
    _suptitle(fig, f"Approximate posterior: {len(calibration.accepted)} accepted of "
                   f"{len(calibration.draws):,} prior draws (x-axis = prior range)", y=1.1)
    return _finish(fig, path)


# --------------------------------------------------------------------------- experiments
def plot_interventions(table: pd.DataFrame, path=None, school_size: int = 763):
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(table) + 1.4))
    y = np.arange(len(table))[::-1]
    ax.hlines(y, table["boys_ill_low"], table["boys_ill_high"], color=SERIES[0], linewidth=2)
    _dot(ax, table["boys_ill_mean"], y, SERIES[0])
    ax.axvline(table["boys_ill_mean"].iloc[0], color=INK_MUTED, linewidth=1)
    for yi, (_, row) in zip(y, table.iterrows()):
        ax.annotate(f"mean {row['boys_ill_mean']:.0f}  ·  major outbreak in {row['major_outbreak_probability']:.0%} of runs",
                    (school_size, yi), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=INK_SECONDARY, annotation_clip=False)
    ax.set_yticks(y, table["intervention"])
    ax.set_xlim(0, school_size)
    ax.set_xlabel(f"Boys who fell ill (of {school_size}): mean and 90% interval across posterior draws and seeds")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Interventions in the calibrated boarding-school model")
    return _finish(fig, path)


def plot_capacity(table: pd.DataFrame, path=None):
    fig, (deaths, attack) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.25})
    for (policy, group), color in zip(table.groupby("policy", sort=False), SERIES):
        deaths.fill_between(group["beds_per_1000"], group["deaths_low"], group["deaths_high"], color=color,
                            alpha=0.18, linewidth=0)
        deaths.plot(group["beds_per_1000"], group["deaths_median"], color=color, marker="o", markersize=6,
                    markeredgecolor=SURFACE, label=policy)
        attack.plot(group["beds_per_1000"], group["attack_rate_median"], color=color, marker="o", markersize=6,
                    markeredgecolor=SURFACE, label=policy)
    for ax in (deaths, attack):
        ax.set_xscale("log")
        ax.set_xticks(table["beds_per_1000"].unique(), [f"{v:g}" for v in table["beds_per_1000"].unique()])
        ax.minorticks_off()
        ax.set_xlabel("Regular hospital beds per 1,000 agents")
    deaths.set_title("Deaths by day 200 (median and 90% interval)")
    deaths.set_ylabel("Deaths")
    deaths.yaxis.set_major_formatter(compact)
    attack.set_title("Attack rate (median)")
    attack.yaxis.set_major_formatter(percent)
    attack.set_ylim(0.6, 1.02)
    deaths.legend(loc="upper center", bbox_to_anchor=(1.1, -0.16), ncol=2)
    _suptitle(fig, "Lower respiratory scenario: hospital capacity and the timing of isolation", y=1.04)
    return _finish(fig, path)
