from ..dynamic_env import Traffic
from .agent import QLearningAgent
import numpy as np


num_episodes = 5
max_steps = 5000
log_interval = 100


def train():
    agent = QLearningAgent(
        num_actions=2,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
    )
    env = Traffic(
        use_gui=True,
        step_length=0.1,
        delay=100,
        min_green_steps=50,
        max_steps=5000,
    )

    all_rewards = []
    all_queues = []
    all_episode_rewards = []

    print(f"\n{'=' * 70}")
    print(f"  Training Agent: {agent.name}")
    print(f"  Episodes: {num_episodes} | Max Steps/Episode: {max_steps}")
    print(f"{'=' * 70}")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_queues = []
        step_rewards = []

        print(f"\n  --- Episode {episode + 1}/{num_episodes} ---")

        for step in range(max_steps):
            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            episode_reward += reward
            episode_queues.append(info["total_queue"])
            step_rewards.append(reward)

            if (step + 1) % log_interval == 0:
                avg_queue = np.mean(episode_queues[-log_interval:])
                avg_reward = np.mean(step_rewards[-log_interval:])
                stats = ""
                if hasattr(agent, "get_stats"):
                    s = agent.get_stats()
                    eps = s.get("epsilon", "N/A")
                    if isinstance(eps, float):
                        stats = f" | ε: {eps:.4f}"

                print(
                    f"    Step {step + 1:5d} | "
                    f"Avg Queue: {avg_queue:5.1f} | "
                    f"Avg Reward: {avg_reward:6.2f} | "
                    f"Phase: {info['phase']}"
                    f"{stats}"
                )

            state = next_state

            if done:
                break

        all_episode_rewards.append(episode_reward)
        all_rewards.extend(step_rewards)
        all_queues.extend(episode_queues)

        avg_q = np.mean(episode_queues) if episode_queues else 0
        print(f"\n  Episode {episode + 1} Summary:")
        print(f"    Total Reward: {episode_reward:.2f}")
        print(f"    Avg Queue Length: {avg_q:.2f}")
        print(f"    Steps: {step + 1}")

    env.close()

    print(
        {
            "agent_name": agent.name,
            "episode_rewards": all_episode_rewards,
            "step_rewards": all_rewards,
            "step_queues": all_queues,
        }
    )
