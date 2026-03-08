"""
Training loop for Multi-Agent Actor-Critic Traffic Signal Control.

Each intersection in the SUMO network is controlled by an independent
Actor-Critic agent.  All agents share the same simulation time-step
but maintain separate weights, traces, and epsilon schedules.

Usage:
    from src.multi_agent.actor_critic.train import train
    train()
"""

import numpy as np

from .agent import MultiAgentActorCritic
from .env import MultiAgentTraffic

# ── Hyper-parameters ──────────────────────────────────────────────────────────

#: Total simulated days (each day = one episode)
NUM_EPISODES = 20

#: Max simulation steps per episode
MAX_STEPS = 5000

#: Console log every N steps
LOG_INTERVAL = 100

#: Save version tag
VERSION = 1

# ── Agent configuration ───────────────────────────────────────────────────────

APPROX = "rbf"  # 'rbf' | 'tile'
N_RBF = 5
N_TILINGS = 8
TILES_PER_DIM = 4

# Rough bounds for each state dimension:
#   [queue_0 .. queue_N (0-20 vehicles), phase (0-3)]
#   Actual state_size is determined dynamically per TLS,
#   but bounds are replicated to match.
QUEUE_LOW = 0
QUEUE_HIGH = 20
PHASE_LOW = 0
PHASE_HIGH = 3

# 3-phase epsilon schedule (paper Table 2)
PHASE_BOUNDARIES = [6, 13]


# ─────────────────────────────────────────────────────────────────────────────


def _make_bounds(state_size: int):
    """Create state_low / state_high lists for a given state dimension."""
    state_low = [QUEUE_LOW] * (state_size - 1) + [PHASE_LOW]
    state_high = [QUEUE_HIGH] * (state_size - 1) + [PHASE_HIGH]
    return state_low, state_high


def train():
    """Run multi-agent Actor-Critic training on the SUMO traffic network."""

    # ── Environment ───────────────────────────────────────────────────────
    env = MultiAgentTraffic(
        use_gui=True,
        step_length=1.0,
        delay=0,
        min_green_steps=5,
        max_steps=MAX_STEPS,
    )

    num_agents = len(env.tls_ids)

    # Use the *maximum* state size across all intersections so every agent
    # shares the same feature-map dimensionality. Agents whose actual
    # detector count is smaller will pad with zeros.
    max_state_size = max(env.state_sizes.values())
    state_low, state_high = _make_bounds(max_state_size)

    # ── Multi-Agent Manager ───────────────────────────────────────────────
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
        alpha_critic=0.1,
        alpha_actor=0.1,
        gamma=0.95,
        lambda_=0.9,
        epsilon_start=0.9,
        epsilon_end=0.0,
        epsilon_decay=0.9991,
    )

    # ── Tracking ──────────────────────────────────────────────────────────
    all_episode_rewards = []
    all_step_rewards = []
    all_step_queues = []

    print(f"\n{'=' * 70}")
    print(f"  Multi-Agent Actor-Critic Training")
    print(f"  Agents         : {num_agents} intersections")
    print(f"  Approx Method  : {APPROX.upper()}")
    print(f"  Max State Size : {max_state_size}")
    print(f"  Episodes       : {NUM_EPISODES}  |  Max steps/ep: {MAX_STEPS}")
    print(f"  TLS IDs        : {env.tls_ids}")
    print(f"{'=' * 70}")

    # ── Training Loop ─────────────────────────────────────────────────────

    for episode in range(NUM_EPISODES):
        # Phase-based epsilon override
        manager.update_epsilon(episode, PHASE_BOUNDARIES)
        manager.reset_traces()

        raw_states = env.reset()

        # Pad states to uniform size
        states = _pad_states(raw_states, max_state_size)

        episode_reward = {tid: 0.0 for tid in env.tls_ids}
        episode_queues = []
        step_deltas = []

        phase_label = (
            "explore-high"
            if episode < PHASE_BOUNDARIES[0]
            else "explore-low" if episode < PHASE_BOUNDARIES[1] else "exploit"
        )

        stats = manager.get_stats()
        print(
            f"\n  --- Episode {episode + 1}/{NUM_EPISODES}  "
            f"[{phase_label}]  avg-ε={stats['avg_epsilon']:.4f} ---"
        )

        for step in range(MAX_STEPS):
            actions = manager.select_actions(states)

            raw_next_states, rewards, done, infos = env.step(actions)
            next_states = _pad_states(raw_next_states, max_state_size)

            td_errors = manager.learn(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                done=done,
            )

            # Accumulate metrics
            total_queue_this_step = sum(info["total_queue"] for info in infos.values())
            total_reward_this_step = sum(rewards.values())
            avg_td = float(np.mean([abs(d) for d in td_errors.values()]))

            for tid in env.tls_ids:
                episode_reward[tid] += rewards.get(tid, 0.0)

            episode_queues.append(total_queue_this_step)
            step_deltas.append(avg_td)
            all_step_rewards.append(total_reward_this_step)
            all_step_queues.append(total_queue_this_step)

            # Logging
            if (step + 1) % LOG_INTERVAL == 0:
                avg_queue = np.mean(episode_queues[-LOG_INTERVAL:])
                avg_reward = np.mean(all_step_rewards[-LOG_INTERVAL:])
                avg_delta = np.mean(step_deltas[-LOG_INTERVAL:])
                s = manager.get_stats()
                print(
                    f"    Step {step + 1:5d} | "
                    f"Σ Queue: {avg_queue:6.1f} | "
                    f"Avg Reward: {avg_reward:7.2f} | "
                    f"TD-err: {avg_delta:.4f} | "
                    f"avg-ε: {s['avg_epsilon']:.4f}"
                )

            states = next_states

            if done:
                break

        # Episode summary
        total_ep_reward = sum(episode_reward.values())
        all_episode_rewards.append(total_ep_reward)
        avg_q = np.mean(episode_queues) if episode_queues else 0.0

        print(f"\n  Episode {episode + 1} Summary:")
        print(f"    Total Reward (all agents) : {total_ep_reward:.2f}")
        print(f"    Avg Total Queue          : {avg_q:.2f}")
        print(f"    Steps                    : {step + 1}")

        # Per-agent breakdown
        for tid in env.tls_ids:
            print(f"      Agent {tid:>12s} : reward={episode_reward[tid]:.2f}")

    # ── Cleanup & Save ────────────────────────────────────────────────────
    env.close()

    save_dir = f"models/multi_agent_ac_{APPROX}_v{VERSION}"
    manager.save(save_dir)

    print("\n" + "=" * 70)
    print("  Multi-Agent Training Complete.")
    print(f"  Models saved to : {save_dir}")
    print("=" * 70)

    return {
        "num_agents": num_agents,
        "episode_rewards": all_episode_rewards,
        "step_rewards": all_step_rewards,
        "step_queues": all_step_queues,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pad_states(states: dict, target_size: int) -> dict:
    """
    Pad each state tuple to *target_size* so all agents share the same
    feature-map dimensionality.  Pads with zeros between the queue values
    and the final phase value.
    """
    padded = {}
    for tid, state in states.items():
        s = list(state)
        if len(s) < target_size:
            # Insert zeros before the final phase element
            phase = s[-1]
            queues = s[:-1]
            queues += [0] * (target_size - len(s))
            s = tuple(queues) + (phase,)
        padded[tid] = tuple(s[:target_size])
    return padded
