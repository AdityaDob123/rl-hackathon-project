"""Professional visualization suite for TradeDesk OpenEnv results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Dark professional theme ───────────────────────
_BG = "#0f1117"
_FG = "#e6edf3"
_GRID = "#21262d"
_ACCENT_COLORS = ["#58a6ff", "#7ee787", "#f0883e", "#f778ba", "#d2a8ff"]

matplotlib.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor": _BG,
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _FG,
    "text.color": _FG,
    "xtick.color": _FG,
    "ytick.color": _FG,
    "grid.color": _GRID,
    "grid.alpha": 0.4,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": _GRID,
    "legend.labelcolor": _FG,
    "font.family": "sans-serif",
    "font.size": 11,
})
sns.set_style("darkgrid", rc={
    "axes.facecolor": _BG,
    "figure.facecolor": _BG,
    "grid.color": _GRID,
})


def ensure_output_dir(path: str = "outputs") -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _flatten_tasks(summary: Dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"task_id": task["task_id"], "difficulty": task["difficulty"], "score": task["final_score"]}
            for task in summary["tasks"]
        ]
    )


def _flatten_steps(summary: Dict) -> pd.DataFrame:
    rows: List[Dict] = []
    for task in summary["tasks"]:
        for step in task["steps"]:
            rows.append(
                {
                    "task_id": task["task_id"],
                    "difficulty": task["difficulty"],
                    "step_index": step["step_index"],
                    "score": step["final_score"],
                    "confidence": step["action"]["confidence"],
                    "reward": step["reward"]["value"],
                    "action_type": step["action"]["action_type"],
                    **{f"component_{k}": v for k, v in step["reward"]["components"].items()},
                }
            )
    return pd.DataFrame(rows)


def _short_task_name(task_id: str) -> str:
    """Convert 'easy_signal_detection' → 'Signal Detection'."""
    parts = task_id.split("_")
    return " ".join(p.capitalize() for p in parts[1:]) if len(parts) > 1 else task_id.replace("_", " ").title()


# ── 1. Score Bar Chart ──────────────────────────

def save_score_bar(summary: Dict, output_dir: str = "outputs") -> str:
    out = ensure_output_dir(output_dir)
    df = _flatten_tasks(summary)
    df["label"] = df["task_id"].apply(_short_task_name)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {d: c for d, c in zip(["easy", "medium", "hard"], _ACCENT_COLORS[:3])}
    bars = ax.bar(
        df["label"],
        df["score"],
        color=[colors.get(d, _ACCENT_COLORS[0]) for d in df["difficulty"]],
        edgecolor="none",
        width=0.55,
        zorder=3,
    )

    # Value labels on bars
    for bar, score in zip(bars, df["score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
            color=_FG,
        )

    ax.set_title("Final Score by Task", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=0)

    # Difficulty legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=colors[d], label=d.capitalize()) for d in colors]
    ax.legend(handles=legend_items, loc="upper left", framealpha=0.7)

    fig.tight_layout()
    path = out / "task_scores.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ── 2. Reward Breakdown ────────────────────────

def save_reward_breakdown(summary: Dict, output_dir: str = "outputs") -> str:
    out = ensure_output_dir(output_dir)
    rows: List[Dict] = []
    for task in summary["tasks"]:
        for step in task["steps"]:
            for key, value in step["reward"]["components"].items():
                rows.append(
                    {
                        "task": _short_task_name(task["task_id"]),
                        "step_index": step["step_index"],
                        "component": key.replace("_", " ").title(),
                        "value": value,
                    }
                )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    palette = sns.color_palette(_ACCENT_COLORS, n_colors=df["task"].nunique())
    sns.barplot(data=df, x="component", y="value", hue="task", ax=ax, palette=palette, edgecolor="none")
    ax.set_title("Reward Component Breakdown", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Contribution", fontsize=12)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Task", framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    path = out / "reward_breakdown.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ── 3. Confidence vs Score ─────────────────────

def save_confidence_vs_score(summary: Dict, output_dir: str = "outputs") -> str:
    out = ensure_output_dir(output_dir)
    df = _flatten_steps(summary)
    diff_colors = {"easy": _ACCENT_COLORS[0], "medium": _ACCENT_COLORS[1], "hard": _ACCENT_COLORS[2]}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for diff, color in diff_colors.items():
        subset = df[df["difficulty"] == diff]
        ax.scatter(
            subset["confidence"],
            subset["score"],
            c=color,
            s=160,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=diff.capitalize(),
            zorder=3,
        )
    ax.set_title("Confidence vs Step Score", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Submitted Confidence", fontsize=12)
    ax.set_ylabel("Step Score", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.1)
    ax.legend(framealpha=0.7)
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    path = out / "confidence_vs_score.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ── 4. Reward Trajectory ──────────────────────

def save_reward_trajectory(summary: Dict, output_dir: str = "outputs") -> str:
    out = ensure_output_dir(output_dir)
    df = _flatten_steps(summary)
    df["task_label"] = df["task_id"].apply(_short_task_name)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {_short_task_name(t): c for t, c in zip(df["task_id"].unique(), _ACCENT_COLORS)}
    sns.lineplot(
        data=df,
        x="step_index",
        y="reward",
        hue="task_label",
        marker="o",
        ax=ax,
        palette=palette,
        linewidth=2.5,
        markersize=10,
    )
    ax.set_title("Reward Trajectory by Task", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Step Index", fontsize=12)
    ax.set_ylabel("Dense Reward", fontsize=12)
    ax.legend(title="Task", framealpha=0.7)
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    path = out / "reward_trajectory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ── 5. Radar Chart (NEW) ──────────────────────

def save_radar_chart(summary: Dict, output_dir: str = "outputs") -> str:
    """Radar chart comparing reward components across tasks."""
    out = ensure_output_dir(output_dir)

    # Aggregate components per task
    task_components: Dict[str, Dict[str, float]] = {}
    for task in summary["tasks"]:
        task_label = _short_task_name(task["task_id"])
        agg: Dict[str, float] = {}
        for step in task["steps"]:
            for key, val in step["reward"]["components"].items():
                agg[key] = agg.get(key, 0) + val
        # Average over steps
        n_steps = len(task["steps"])
        task_components[task_label] = {k: v / n_steps for k, v in agg.items()}

    categories = list(next(iter(task_components.values())).keys())
    cat_labels = [c.replace("_", " ").title() for c in categories]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)

    for idx, (task_label, comps) in enumerate(task_components.items()):
        values = [comps[c] for c in categories]
        values += values[:1]
        color = _ACCENT_COLORS[idx % len(_ACCENT_COLORS)]
        ax.fill(angles, values, alpha=0.2, color=color)
        ax.plot(angles, values, linewidth=2.5, label=task_label, color=color, marker="o", markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=10, color=_FG)
    ax.set_title("Reward Profile Comparison", fontsize=16, fontweight="bold", pad=24, color=_FG)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.7)

    # Style the grid
    ax.spines["polar"].set_color(_GRID)
    ax.tick_params(colors=_FG)
    for label in ax.get_yticklabels():
        label.set_color(_FG)

    fig.tight_layout()
    path = out / "reward_radar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ── 6. Action Heatmap (NEW) ──────────────────

def save_action_heatmap(summary: Dict, output_dir: str = "outputs") -> str:
    """Heatmap: action types × difficulty → average score."""
    out = ensure_output_dir(output_dir)
    df = _flatten_steps(summary)

    pivot = df.pivot_table(values="score", index="action_type", columns="difficulty", aggfunc="mean").fillna(0)
    # Reorder columns
    col_order = [c for c in ["easy", "medium", "hard"] if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = sns.color_palette("blend:#0f1117,#58a6ff", as_cmap=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=2,
        linecolor=_BG,
        ax=ax,
        cbar_kws={"label": "Avg Score", "shrink": 0.8},
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    ax.set_title("Action × Difficulty Score Matrix", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Difficulty", fontsize=12)
    ax.set_ylabel("Action Type", fontsize=12)
    ax.tick_params(rotation=0)
    fig.tight_layout()
    path = out / "action_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ── Orchestrator ──────────────────────────────

def save_all_plots(summary: Dict, output_dir: str = "outputs") -> List[str]:
    return [
        save_score_bar(summary, output_dir=output_dir),
        save_reward_breakdown(summary, output_dir=output_dir),
        save_confidence_vs_score(summary, output_dir=output_dir),
        save_reward_trajectory(summary, output_dir=output_dir),
        save_radar_chart(summary, output_dir=output_dir),
        save_action_heatmap(summary, output_dir=output_dir),
    ]
