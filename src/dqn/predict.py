"""
Prediction / Inference for DQN Traffic Signal Controller.

Loads a trained DQN model from disk and runs the SUMO simulation in
pure exploitation mode (ε = 0) to evaluate signal control performance.

Usage:
    from src.dqn.predict import predict
    results = predict()                           # default model path
    results = predict(model_path="models/dqn_v2.pth")
"""

import numpy as np
from ..dynamic_env import Traffic
from .agent import DQNAgent


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/dqn_v1.pth"
MAX_STEPS = 5000
LOG_INTERVAL = 100


# ─────────────────────────────────────────────────────────────────────────────


def predict(
    model_path: str = DEFAULT_MODEL_PATH,
    max_steps: int = MAX_STEPS,
    use_gui: bool = True,
    num_episodes: int = 1,
):
    """
    Run a trained DQN agent on the SUMO traffic environment.

    Parameters
    ----------
    model_path : str
        Path to the saved .pth model file.
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
    # Create env first and probe the real state size before building the agent
    env = Traffic(
        use_gui=use_gui,
        step_length=0.1,
        delay=100,
        min_green_steps=50,
        max_steps=max_steps,
    )
    sample_state = env.reset()
    real_state_size = len(sample_state)

    # ── Load Agent ────────────────────────────────────────────────────────
    agent = DQNAgent(
        state_size=real_state_size,
        num_actions=Traffic.NUM_ACTIONS,
        epsilon_start=0.0,  # pure exploitation
        epsilon_end=0.0,
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # ensure greedy

    print(f"\n{'=' * 70}")
    print("  DQN — Prediction Mode")
    print(f"  Model Path     : {model_path}")
    print(f"  State Size     : {real_state_size}")
    print(f"  Episodes       : {num_episodes}  |  Max steps/ep: {max_steps}")
    print(f"{'=' * 70}")

    # ── Evaluation Loop ──────────────────────────────────────────────────

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

            # Logging
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

        # ── Episode Summary ───────────────────────────────────────────────
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

    # ── Final Summary ─────────────────────────────────────────────────────
    env.close()

    avg_total_reward = np.mean([r["total_reward"] for r in all_episode_results])
    avg_total_queue = np.mean([r["avg_queue"] for r in all_episode_results])

    print(f"\n{'=' * 70}")
    print(f"  DQN Prediction Complete — {num_episodes} episode(s)")
    print(f"  Avg Total Reward : {avg_total_reward:.2f}")
    print(f"  Avg Queue Length : {avg_total_queue:.2f}")
    print(f"{'=' * 70}")

    return {
        "agent_name": "DQN",
        "episodes": all_episode_results,
        "avg_total_reward": float(avg_total_reward),
        "avg_total_queue": float(avg_total_queue),
    }
