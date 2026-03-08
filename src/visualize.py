"""
Visualization utilities for RSL test results.

Generates and saves matplotlib charts to the test output directory.
"""

import os
import numpy as np


def _rolling_mean(values, window=50):
    """Simple rolling mean using numpy convolution."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def save_visualizations(results: dict, output_dir: str):
    """
    Generate and save evaluation charts to *output_dir/visualizations/*.

    Charts produced (where data is available):
      1. Queue length over time  (per episode + rolling mean)
      2. Reward over time        (per episode + rolling mean)
      3. Episode summary bar     (total reward and avg queue per episode)
      4. Phase switches bar      (per episode)

    Parameters
    ----------
    results : dict   Output from any predict() function.
    output_dir : str Target directory (charts go to output_dir/visualizations/).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping visualizations.")
        return

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    episodes = results.get("episodes", [])
    agent_name = results.get("agent_name", "Agent")
    num_ep = len(episodes)

    COLORS = ["#4C9BE8", "#E86B4C", "#4CE87A", "#E8D44C"]
    STYLE = {"linewidth": 1.2, "alpha": 0.4}

    # ── 1 & 2. Per-step time-series (queue + reward) ──────────────────────
    has_steps = episodes and "step_queues" in episodes[0]
    if has_steps:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        fig.suptitle(
            f"{agent_name} — Step-by-Step Evaluation", fontsize=14, fontweight="bold"
        )

        for i, ep in enumerate(episodes):
            color = COLORS[i % len(COLORS)]
            label = f"Ep {ep['episode']}"

            queues = ep["step_queues"]
            rewards = ep["step_rewards"]
            xs = list(range(1, len(queues) + 1))

            axes[0].plot(xs, queues, color=color, label=label, **STYLE)
            axes[0].plot(xs, _rolling_mean(queues), color=color, linewidth=2.0)

            axes[1].plot(xs, rewards, color=color, label=label, **STYLE)
            axes[1].plot(xs, _rolling_mean(rewards), color=color, linewidth=2.0)

        axes[0].set_ylabel("Total Queue Length")
        axes[0].set_xlabel("Step")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("Reward")
        axes[1].set_xlabel("Step")
        axes[1].axhline(0, color="white", linewidth=0.5, linestyle="--")
        axes[1].legend(loc="lower left", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(viz_dir, "step_timeseries.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    • {path}")

    # ── 3. Episode summary bar chart ──────────────────────────────────────
    if episodes:
        ep_nums = [e["episode"] for e in episodes]
        tot_rewards = [e["total_reward"] for e in episodes]
        avg_queues = [e["avg_queue"] for e in episodes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"{agent_name} — Episode Summary", fontsize=14, fontweight="bold")

        x = np.arange(num_ep)
        bars1 = ax1.bar(
            x, tot_rewards, color=COLORS[:num_ep] if num_ep <= 4 else COLORS[0]
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Ep {n}" for n in ep_nums])
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Total Reward per Episode")
        ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax1.bar_label(bars1, fmt="%.0f", padding=3, fontsize=8)
        ax1.grid(True, axis="y", alpha=0.3)

        bars2 = ax2.bar(
            x, avg_queues, color=COLORS[:num_ep] if num_ep <= 4 else COLORS[1]
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Ep {n}" for n in ep_nums])
        ax2.set_ylabel("Avg Queue Length")
        ax2.set_title("Average Queue per Episode")
        ax2.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(viz_dir, "episode_summary.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    • {path}")

    # ── 4. Phase switches bar ─────────────────────────────────────────────
    if episodes and "switches" in episodes[0]:
        switches = [e["switches"] for e in episodes]
        ep_nums = [e["episode"] for e in episodes]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(
            f"{agent_name} — Phase Switches per Episode", fontsize=13, fontweight="bold"
        )
        bars = ax.bar(range(num_ep), switches, color="#A44CE8")
        ax.set_xticks(range(num_ep))
        ax.set_xticklabels([f"Ep {n}" for n in ep_nums])
        ax.set_ylabel("# Phase Switches")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(viz_dir, "phase_switches.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    • {path}")
