"""
Prediction / Inference for Multi-Agent Actor-Critic Traffic Signal Control.

Loads trained agent weights from disk and runs a SUMO simulation where
every intersection is controlled by its corresponding Actor-Critic agent
in **pure exploitation mode** (ε = 0).

Collects per-intersection and aggregate performance metrics, including
queue lengths, waiting times, rewards, and throughput.

Usage:
    from src.multi_agent.actor_critic.predict import predict
    results = predict()                       # default model dir
    results = predict(model_dir="models/multi_agent_ac_rbf_v2")
"""

import numpy as np

from .agent import MultiAgentActorCritic
from .env import MultiAgentTraffic

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "models/multi_agent_ac_rbf_v1"
MAX_STEPS = 5000
LOG_INTERVAL = 100

# Feature-map settings (must match training)
APPROX = "rbf"
N_RBF = 5
N_TILINGS = 8
TILES_PER_DIM = 4
QUEUE_LOW = 0
QUEUE_HIGH = 20
PHASE_LOW = 0
PHASE_HIGH = 3


# ─────────────────────────────────────────────────────────────────────────────


def _make_bounds(state_size: int):
    state_low = [QUEUE_LOW] * (state_size - 1) + [PHASE_LOW]
    state_high = [QUEUE_HIGH] * (state_size - 1) + [PHASE_HIGH]
    return state_low, state_high


def _pad_states(states: dict, target_size: int) -> dict:
    """Pad each state tuple to *target_size*."""
    padded = {}
    for tid, state in states.items():
        s = list(state)
        if len(s) < target_size:
            phase = s[-1]
            queues = s[:-1]
            queues += [0] * (target_size - len(s))
            s = tuple(queues) + (phase,)
        padded[tid] = tuple(s[:target_size])
    return padded


# ─────────────────────────────────────────────────────────────────────────────


def predict(
    model_dir: str = DEFAULT_MODEL_DIR,
    max_steps: int = MAX_STEPS,
    use_gui: bool = True,
    num_episodes: int = 1,
):
    """
    Run trained multi-agent AC models on the SUMO network.

    Parameters
    ----------
    model_dir : str
        Directory containing agent .npz weight files.
    max_steps : int
        Maximum simulation steps per episode.
    use_gui : bool
        Whether to launch the SUMO GUI.
    num_episodes : int
        Number of evaluation episodes to run.

    Returns
    -------
    results : dict
        Aggregate performance metrics.
    """

    # ── Environment ───────────────────────────────────────────────────────
    env = MultiAgentTraffic(
        use_gui=use_gui,
        step_length=1.0,
        delay=50,  # slight GUI delay so humans can watch
        min_green_steps=5,
        max_steps=max_steps,
    )

    num_agents = len(env.tls_ids)
    max_state_size = max(env.state_sizes.values())
    state_low, state_high = _make_bounds(max_state_size)

    # ── Load Agents ───────────────────────────────────────────────────────
    manager = MultiAgentActorCritic(
        tls_ids=env.tls_ids,
        state_size=max_state_size,
        num_actions=MultiAgentTraffic.NUM_ACTIONS,
        approx=APPROX,
        n_rbf=N_RBF,
        n_tilings=N_TILINGS,
        tiles_per_dim=TILES_PER_DIM,
        state_low=state_low,
        state_high=state_high,
        epsilon_start=0.0,  # pure exploitation
        epsilon_end=0.0,
    )
    manager.load(model_dir)

    # Force all epsilons to zero (greedy)
    for agent in manager.agents.values():
        agent.epsilon = 0.0

    print(f"\n{'=' * 70}")
    print(f"  Multi-Agent Actor-Critic — Prediction Mode")
    print(f"  Model Dir      : {model_dir}")
    print(f"  Agents         : {num_agents} intersections")
    print(f"  Episodes       : {num_episodes}  |  Max steps/ep: {max_steps}")
    print(f"  TLS IDs        : {env.tls_ids}")
    print(f"{'=' * 70}")

    # ── Evaluation Loop ──────────────────────────────────────────────────

    all_episode_results = []

    for ep in range(num_episodes):
        raw_states = env.reset()
        states = _pad_states(raw_states, max_state_size)

        ep_rewards = {tid: 0.0 for tid in env.tls_ids}
        ep_queues = []
        ep_actions = {tid: [] for tid in env.tls_ids}

        print(f"\n  --- Episode {ep + 1}/{num_episodes} ---")

        for step in range(max_steps):
            actions = manager.select_actions(states)

            raw_next_states, rewards, done, infos = env.step(actions)
            next_states = _pad_states(raw_next_states, max_state_size)

            # Track metrics
            total_queue = sum(info["total_queue"] for info in infos.values())
            ep_queues.append(total_queue)

            for tid in env.tls_ids:
                ep_rewards[tid] += rewards.get(tid, 0.0)
                ep_actions[tid].append(actions.get(tid, 0))

            # Logging
            if (step + 1) % LOG_INTERVAL == 0:
                avg_queue = np.mean(ep_queues[-LOG_INTERVAL:])
                avg_reward = np.mean([sum(rewards.values())] * 1)  # current step
                # Per-agent phase snapshot
                phases = {tid: infos[tid]["phase"] for tid in env.tls_ids}
                print(
                    f"    Step {step + 1:5d} | "
                    f"Σ Queue: {avg_queue:6.1f} | "
                    f"Reward: {sum(rewards.values()):7.2f} | "
                    f"Phases: {phases}"
                )

            states = next_states

            if done:
                break

        # ── Episode Summary ───────────────────────────────────────────────
        total_reward = sum(ep_rewards.values())
        avg_q = np.mean(ep_queues) if ep_queues else 0.0
        switch_counts = {tid: sum(ep_actions[tid]) for tid in env.tls_ids}

        print(f"\n  Episode {ep + 1} Summary:")
        print(f"    Total Reward (all agents) : {total_reward:.2f}")
        print(f"    Avg Total Queue           : {avg_q:.2f}")
        print(f"    Steps Completed           : {step + 1}")
        print(f"    Phase Switches per Agent  :")
        for tid in env.tls_ids:
            print(
                f"      {tid:>12s} : "
                f"reward={ep_rewards[tid]:8.2f}  "
                f"switches={switch_counts[tid]}"
            )

        all_episode_results.append(
            {
                "episode": ep + 1,
                "total_reward": total_reward,
                "avg_queue": avg_q,
                "steps": step + 1,
                "per_agent_rewards": dict(ep_rewards),
                "per_agent_switches": dict(switch_counts),
            }
        )

    # ── Final Summary ─────────────────────────────────────────────────────
    env.close()

    avg_total_reward = np.mean([r["total_reward"] for r in all_episode_results])
    avg_total_queue = np.mean([r["avg_queue"] for r in all_episode_results])

    print(f"\n{'=' * 70}")
    print(f"  Prediction Complete — {num_episodes} episode(s)")
    print(f"  Avg Total Reward : {avg_total_reward:.2f}")
    print(f"  Avg Total Queue  : {avg_total_queue:.2f}")
    print(f"{'=' * 70}")

    return {
        "num_agents": num_agents,
        "episodes": all_episode_results,
        "avg_total_reward": float(avg_total_reward),
        "avg_total_queue": float(avg_total_queue),
    }
