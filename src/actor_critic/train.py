"""
Training loop for Actor-Critic Adaptive Traffic Signal Controller (A-CAT).

Follows the same structure as qlearning_single_agent/train.py and dqn/train.py.

Epsilon schedule mirrors the paper's 3-phase training:
  Phase 1 (days 1-6):   high exploration (ε decays from 0.9 → 0.1)
  Phase 2 (days 7-13):  low exploration (ε = 0.1, epsilon_decay ≈ 1.0)
  Phase 3 (days 14-20): pure exploitation (ε = 0.0)

Usage:
    from src.actor_critic.train import train
    train()
"""

import numpy as np
from ..dynamic_env import Traffic
from .agent import ActorCriticAgent


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

# Change to 'tile' and adjust n_rbf / n_tilings to compare variants.
APPROX = "rbf"  # 'rbf' | 'tile'
N_RBF = 5  # Paper winner: 5 centres per dimension
N_TILINGS = 8  # Used when APPROX='tile'
TILES_PER_DIM = 4

# Rough bounds for each state dimension:
#   [queue_0 .. queue_7 (0-20 vehicles), phase (0-3)]
STATE_LOW = [0] * 8 + [0]
STATE_HIGH = [20] * 8 + [3]

# 3-phase epsilon schedule (paper Table 2)
#   Phase 1 → epsilon decays from 0.9 to 0.1 over first 6 episodes
#   Phase 2 → epsilon stays at 0.1 for episodes 7-13
#   Phase 3 → epsilon = 0 for episodes 14-20
PHASE_BOUNDARIES = [6, 13]  # end of phase 1, end of phase 2


def _update_epsilon(agent: ActorCriticAgent, episode: int):
    """Hard-set epsilon boundaries between training phases."""
    if episode < PHASE_BOUNDARIES[0]:
        pass  # epsilon is decaying naturally in this phase
    elif episode < PHASE_BOUNDARIES[1]:
        # Clamp to 0.1 at start of phase 2
        if agent.epsilon < 0.1:
            agent.epsilon = 0.1
    else:
        # Pure exploitation
        agent.epsilon = 0.0


# ─────────────────────────────────────────────────────────────────────────────


def train():
    """Run Actor-Critic training on the SUMO traffic environment."""

    agent = ActorCriticAgent(
        state_size=Traffic.STATE_SIZE,
        num_actions=Traffic.NUM_ACTIONS,
        approx=APPROX,
        n_rbf=N_RBF,
        n_tilings=N_TILINGS,
        tiles_per_dim=TILES_PER_DIM,
        state_low=STATE_LOW,
        state_high=STATE_HIGH,
        alpha_critic=0.1,
        alpha_actor=0.1,
        gamma=0.95,
        lambda_=0.9,
        epsilon_start=0.9,
        epsilon_end=0.0,
        epsilon_decay=0.9991,  # ~0.9→0.1 over 6*5000=30000 steps
    )

    env = Traffic(
        use_gui=True,
        step_length=0.1,
        delay=100,
        min_green_steps=50,
        max_steps=MAX_STEPS,
    )

    all_rewards = []
    all_queues = []
    all_episode_rewards = []

    print(f"\n{'=' * 70}")
    print(f"  Training Agent : {agent.name}")
    print(
        f"  Approx Method  : {APPROX.upper()}  |  Feature size: {agent.feature_map.feature_size}"
    )
    print(f"  Episodes       : {NUM_EPISODES}  |  Max steps/ep: {MAX_STEPS}")
    print(f"{'=' * 70}")

    for episode in range(NUM_EPISODES):
        # Apply phase-based epsilon override before each episode
        _update_epsilon(agent, episode)

        state = env.reset()
        episode_reward = 0.0
        episode_queues = []
        step_rewards = []
        step_deltas = []

        phase_label = (
            "explore-high"
            if episode < PHASE_BOUNDARIES[0]
            else "explore-low"
            if episode < PHASE_BOUNDARIES[1]
            else "exploit"
        )
        print(
            f"\n  --- Episode {episode + 1}/{NUM_EPISODES}  [{phase_label}]  ε={agent.epsilon:.4f} ---"
        )

        for step in range(MAX_STEPS):
            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            delta = agent.learn(state, action, reward, next_state, done)

            episode_reward += reward
            episode_queues.append(info["total_queue"])
            step_rewards.append(reward)
            step_deltas.append(abs(delta))

            if (step + 1) % LOG_INTERVAL == 0:
                avg_queue = np.mean(episode_queues[-LOG_INTERVAL:])
                avg_reward = np.mean(step_rewards[-LOG_INTERVAL:])
                avg_delta = np.mean(step_deltas[-LOG_INTERVAL:])
                s = agent.get_stats()
                print(
                    f"    Step {step + 1:5d} | "
                    f"Avg Queue: {avg_queue:5.1f} | "
                    f"Avg Reward: {avg_reward:6.2f} | "
                    f"TD-err: {avg_delta:.4f} | "
                    f"Phase: {info['phase']} | "
                    f"ε: {s['epsilon']:.4f}"
                )

            state = next_state

            if done:
                break

        all_episode_rewards.append(episode_reward)
        all_rewards.extend(step_rewards)
        all_queues.extend(episode_queues)

        avg_q = np.mean(episode_queues) if episode_queues else 0.0
        print(f"\n  Episode {episode + 1} Summary:")
        print(f"    Total Reward  : {episode_reward:.2f}")
        print(f"    Avg Queue Len : {avg_q:.2f}")
        print(f"    Steps         : {step + 1}")

    env.close()

    save_path = f"models/actor_critic_{APPROX}_rbf{N_RBF}_v{VERSION}.npz"
    agent.save(save_path)

    print("\n" + "=" * 70)
    print("  Training complete.")
    print(f"  Model saved to : {save_path}")
    print("=" * 70)

    return {
        "agent_name": agent.name,
        "episode_rewards": all_episode_rewards,
        "step_rewards": all_rewards,
        "step_queues": all_queues,
    }
