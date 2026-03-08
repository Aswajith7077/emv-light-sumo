"""

Usage
-----
    python main.py train   <agent>
    python main.py predict <agent>
    python main.py test    <agent>

Agents
------
    qlearning               Q-Learning (single intersection)
    dqn                     Deep Q-Network (single intersection)
    actor-critic            Actor-Critic (single intersection)
    multi-agent_actor-critic  Multi-Agent Actor-Critic (all intersections)

Examples
--------
    python main.py train dqn
    python main.py predict qlearning
    python main.py test multi-agent_actor-critic
"""

import sys
import os
import json
import numpy as np


# ── Agent registry ────────────────────────────────────────────────────────────

AGENTS = {
    "qlearning": {
        "train": "src.qlearning_single_agent.train",
        "predict": "src.qlearning_single_agent.predict",
        "label": "Q-Learning",
    },
    "dqn": {
        "train": "src.dqn.train",
        "predict": "src.dqn.predict",
        "label": "Deep Q-Network (DQN)",
    },
    "actor-critic": {
        "train": "src.actor_critic.train",
        "predict": "src.actor_critic.predict",
        "label": "Actor-Critic",
    },
    "multi-agent_actor-critic": {
        "train": "src.multi_agent.actor_critic.train",
        "predict": "src.multi_agent.actor_critic.predict",
        "label": "Multi-Agent Actor-Critic",
    },
}

VALID_MODES = ["train", "predict", "test"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _print_usage():
    print(__doc__)
    sys.exit(1)


def _import_function(module_path: str, func_name: str):
    """Dynamically import *func_name* from *module_path*."""
    from importlib import import_module

    mod = import_module(module_path)
    return getattr(mod, func_name)


def _save_results(results: dict, output_dir: str):
    """Persist evaluation results as JSON + human-readable summary."""
    os.makedirs(output_dir, exist_ok=True)

    # ── JSON dump ─────────────────────────────────────────────────────────
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)

    # ── Human-readable summary ────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")

        agent_name = results.get("agent_name", results.get("num_agents", "N/A"))
        f.write(f"  Agent            : {agent_name}\n")

        if "avg_total_reward" in results:
            f.write(f"  Avg Total Reward : {results['avg_total_reward']:.2f}\n")
        if "avg_total_queue" in results:
            f.write(f"  Avg Total Queue  : {results['avg_total_queue']:.2f}\n")

        episodes = results.get("episodes", [])
        if episodes:
            f.write(f"\n  Episodes         : {len(episodes)}\n")
            f.write("-" * 60 + "\n")
            for ep in episodes:
                ep_num = ep.get("episode", "?")
                f.write(f"\n  Episode {ep_num}:\n")
                for k, v in ep.items():
                    if k in ("episode", "step_queues", "step_rewards"):
                        continue
                    if isinstance(v, float):
                        f.write(f"    {k:20s}: {v:.2f}\n")
                    elif isinstance(v, dict):
                        f.write(f"    {k}:\n")
                        for kk, vv in v.items():
                            f.write(f"      {kk:16s}: {vv}\n")
                    else:
                        f.write(f"    {k:20s}: {v}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\n  Results saved to: {output_dir}/")
    print(f"    • {json_path}")
    print(f"    • {txt_path}")

    # ── Visualizations ────────────────────────────────────────────────────
    print("\n  Generating visualizations...")
    try:
        from src.visualize import save_visualizations

        save_visualizations(results, output_dir)
    except Exception as e:
        print(f"  [WARN] Visualization failed: {e}")


# ── Mode handlers ─────────────────────────────────────────────────────────────


def handle_train(agent_key: str):
    """Import and run the train() function for the chosen agent."""
    info = AGENTS[agent_key]
    print(f"\n  [TRAIN] {info['label']}\n")
    train_fn = _import_function(info["train"], "train")
    train_fn()


def handle_predict(agent_key: str):
    """Import and run the predict() function for the chosen agent."""
    info = AGENTS[agent_key]
    print(f"\n  [PREDICT] {info['label']}\n")
    predict_fn = _import_function(info["predict"], "predict")
    predict_fn()


def handle_test(agent_key: str):
    """
    Run the predict function in headless mode and save results
    to result/<agent_key>/.
    """
    info = AGENTS[agent_key]
    output_dir = os.path.join("result", agent_key)

    print(f"\n  [TEST] {info['label']}")
    print(f"  Output dir: {output_dir}\n")

    predict_fn = _import_function(info["predict"], "predict")
    results = predict_fn(use_gui=False, num_episodes=3)

    _save_results(results, output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 3:
        _print_usage()

    mode = sys.argv[1].lower()
    agent_key = sys.argv[2].lower()

    if mode not in VALID_MODES:
        print(f"\n  ERROR: Unknown mode '{mode}'.")
        print(f"  Valid modes: {', '.join(VALID_MODES)}")
        _print_usage()

    if agent_key not in AGENTS:
        print(f"\n  ERROR: Unknown agent '{agent_key}'.")
        print(f"  Valid agents: {', '.join(AGENTS.keys())}")
        _print_usage()

    if mode == "train":
        handle_train(agent_key)
    elif mode == "predict":
        handle_predict(agent_key)
    elif mode == "test":
        handle_test(agent_key)


if __name__ == "__main__":
    main()
