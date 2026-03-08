"""
Prediction / Inference for single-agent Actor-Critic Traffic Signal Controller.

Loads trained weights from disk and runs the SUMO simulation in
pure exploitation mode (ε = 0).

Usage:
    from src.actor_critic.predict import predict
    results = predict()
    results = predict(model_path="models/actor_critic_rbf_rbf5_v2.npz")
"""

import numpy as np

from ..dynamic_env import Traffic
from .agent import ActorCriticAgent

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/actor_critic_rbf_rbf5_v1.npz"
MAX_STEPS = 5000
LOG_INTERVAL = 100

APPROX = "rbf"
N_RBF = 5
N_TILINGS = 8
TILES_PER_DIM = 4
STATE_LOW = [0] * 8 + [0]
STATE_HIGH = [20] * 8 + [3]


# ─────────────────────────────────────────────────────────────────────────────


def predict(
    model_path: str = DEFAULT_MODEL_PATH,
    max_steps: int = MAX_STEPS,
    use_gui: bool = True,
    num_episodes: int = 1,
):
    """
    Run a trained Actor-Critic agent on the SUMO traffic environment.

    Parameters
    ----------
    model_path : str
        Path to the saved .npz weights file.
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

    env = Traffic(
        use_gui=use_gui,
        step_length=1.0,
        delay=50,
        min_green_steps=5,
        max_steps=max_steps,
    )
    sample_state = env.reset()
    real_state_size = len(sample_state)

    n_lanes = real_state_size - 1
    state_low = [0] * n_lanes + [0]
    state_high = [20] * n_lanes + [9]

    agent = ActorCriticAgent(
        state_size=real_state_size,
        num_actions=Traffic.NUM_ACTIONS,
        approx=APPROX,
        n_rbf=N_RBF,
        n_tilings=N_TILINGS,
        tiles_per_dim=TILES_PER_DIM,
        state_low=state_low,
        state_high=state_high,
        epsilon_start=0.0,
        epsilon_end=0.0,
    )
    agent.load(model_path)
    agent.epsilon = 0.0

    print(f"\n{'=' * 70}")
    print("  Actor-Critic — Prediction Mode")
    print(f"  Model Path     : {model_path}")
    print(f"  State Size     : {real_state_size}")
    print(f"  Approx         : {APPROX.upper()}")
    print(f"  Episodes       : {num_episodes}  |  Max steps/ep: {max_steps}")
    print(f"{'=' * 70}")

    all_episode_results = []

    for ep in range(num_episodes):
        state = sample_state if ep == 0 else env.reset()

        ep_reward = 0.0
        ep_queues = []
        ep_rewards = []
        ep_switches = 0

        print(f"\n  --- Episode {ep + 1}/{num_episodes} ---")

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            ep_queues.append(info["total_queue"])
            ep_rewards.append(reward)
            ep_switches += action

            if (step + 1) % LOG_INTERVAL == 0:
                avg_queue = np.mean(ep_queues[-LOG_INTERVAL:])
                print(
                    f"    Step {step + 1:5d} | "
                    f"Avg Queue: {avg_queue:5.1f} | "
                    f"Reward: {reward:7.2f} | "
                    f"Phase: {info['phase']}"
                )

            state = next_state
            if done:
                break

        avg_q = np.mean(ep_queues) if ep_queues else 0.0
        print(f"\n  Episode {ep + 1} Summary:")
        print(f"    Total Reward     : {ep_reward:.2f}")
        print(f"    Avg Queue Length : {avg_q:.2f}")
        print(f"    Phase Switches   : {ep_switches}")
        print(f"    Steps            : {step + 1}")

        all_episode_results.append(
            {
                "episode": ep + 1,
                "total_reward": ep_reward,
                "avg_queue": avg_q,
                "switches": ep_switches,
                "steps": step + 1,
                "step_queues": ep_queues,
                "step_rewards": ep_rewards,
            }
        )

    env.close()

    avg_total_reward = np.mean([r["total_reward"] for r in all_episode_results])
    avg_total_queue = np.mean([r["avg_queue"] for r in all_episode_results])

    print(f"\n{'=' * 70}")
    print(f"  Actor-Critic Prediction Complete — {num_episodes} episode(s)")
    print(f"  Avg Total Reward : {avg_total_reward:.2f}")
    print(f"  Avg Queue Length : {avg_total_queue:.2f}")
    print(f"{'=' * 70}")

    return {
        "agent_name": "Actor-Critic",
        "episodes": all_episode_results,
        "avg_total_reward": float(avg_total_reward),
        "avg_total_queue": float(avg_total_queue),
    }
