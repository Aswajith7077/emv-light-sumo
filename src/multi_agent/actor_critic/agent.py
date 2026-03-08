"""
Multi-Agent Actor-Critic for Adaptive Traffic Signal Control.

Provides:
    ActorCriticAgent   – single-intersection AC agent (reuses feature_maps)
    MultiAgentActorCritic – manager that creates one agent per TLS intersection

Each agent maintains its own actor weights W, critic weights θ,
and eligibility traces, and learns from its own local reward signal.
"""

import os
import random
import numpy as np
from ...actor_critic.feature_maps import RBFFeatureMap, TileCodingFeatureMap


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-Intersection Actor-Critic Agent
# ═══════════════════════════════════════════════════════════════════════════════


class ActorCriticAgent:
    """
    One-step TD Actor-Critic with eligibility traces and linear FA.

    Identical to the single-agent version in src/actor_critic/agent.py but
    kept self-contained here so agent hyper-parameters can differ per
    intersection if needed in the future.
    """

    def __init__(
        self,
        agent_id: str,
        state_size: int = 9,
        num_actions: int = 2,
        approx: str = "rbf",
        n_rbf: int = 5,
        n_tilings: int = 8,
        tiles_per_dim: int = 4,
        state_low=None,
        state_high=None,
        alpha_critic: float = 0.1,
        alpha_actor: float = 0.1,
        gamma: float = 0.95,
        lambda_: float = 0.9,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        delta_clip: float = 10.0,
        trace_max_norm: float = 10.0,
        w_max_norm: float = 50.0,
    ):
        self.agent_id = agent_id
        self.state_size = state_size
        self.num_actions = num_actions
        self.approx = approx
        self.alpha_critic = alpha_critic
        self.alpha_actor = alpha_actor
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_steps = 0
        self.delta_clip = delta_clip
        self.trace_max_norm = trace_max_norm
        self.w_max_norm = w_max_norm

        # ── Feature map ───────────────────────────────────────────────────
        if approx == "rbf":
            self.name = f"AC-{agent_id} (RBF-{n_rbf})"
            self.feature_map = RBFFeatureMap(
                state_size=state_size,
                n_rbf=n_rbf,
                state_low=state_low,
                state_high=state_high,
            )
        elif approx == "tile":
            self.name = f"AC-{agent_id} (Tile-{n_tilings}x{tiles_per_dim})"
            self.feature_map = TileCodingFeatureMap(
                state_size=state_size,
                n_tilings=n_tilings,
                tiles_per_dim=tiles_per_dim,
                state_low=state_low,
                state_high=state_high,
            )
        else:
            raise ValueError(
                f"Unknown approx method '{approx}'. Choose 'rbf' or 'tile'."
            )

        fs = self.feature_map.feature_size

        # Critic weights  (value function θᵀφ)
        self.theta = np.zeros(fs, dtype=np.float64)

        # Actor weights   (h(s,a) = W[:,a]ᵀφ)
        self.W = np.zeros((fs, num_actions), dtype=np.float64)

        # Eligibility traces
        self.e_theta = np.zeros(fs, dtype=np.float64)
        self.e_W = np.zeros((fs, num_actions), dtype=np.float64)

    # ── Policy ────────────────────────────────────────────────────────────

    def _preferences(self, phi: np.ndarray) -> np.ndarray:
        """Compute actor preferences h(s, a) = W[:,a]ᵀ φ for all a."""
        return phi @ self.W

    def _softmax_policy(self, phi: np.ndarray) -> np.ndarray:
        """Softmax policy π(a|s) from actor preferences."""
        h = self._preferences(phi)
        h -= h.max()
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def select_action(self, state) -> int:
        """
        ε-greedy action selection over softmax actor policy.

        Parameters
        ----------
        state : tuple or array-like

        Returns
        -------
        action : int
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        phi = self.feature_map(np.array(state, dtype=np.float64))
        probs = self._softmax_policy(phi)
        return int(np.argmax(probs))

    # ── Learning ──────────────────────────────────────────────────────────

    def learn(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> float:
        """
        One-step TD Actor-Critic update with eligibility traces.

        Returns the TD error δ for logging.
        """
        phi = self.feature_map(np.array(state, dtype=np.float64))
        phi_next = self.feature_map(np.array(next_state, dtype=np.float64))

        v_s = float(self.theta @ phi)
        v_s_next = 0.0 if done else float(self.theta @ phi_next)
        delta = reward + self.gamma * v_s_next - v_s

        probs = self._softmax_policy(phi)
        indicator = np.zeros(self.num_actions, dtype=np.float64)
        indicator[action] = 1.0
        grad_log_pi = np.outer(phi, indicator - probs)

        delta = float(np.clip(delta, -self.delta_clip, self.delta_clip))

        # Update eligibility traces
        self.e_theta = self.gamma * self.lambda_ * self.e_theta + phi
        self.e_W = self.gamma * self.lambda_ * self.e_W + grad_log_pi

        # Clip traces
        e_theta_norm = np.linalg.norm(self.e_theta)
        if e_theta_norm > self.trace_max_norm:
            self.e_theta *= self.trace_max_norm / e_theta_norm

        e_W_norm = np.linalg.norm(self.e_W)
        if e_W_norm > self.trace_max_norm:
            self.e_W *= self.trace_max_norm / e_W_norm

        # Weight updates
        self.theta += self.alpha_critic * delta * self.e_theta
        self.W += self.alpha_actor * delta * self.e_W

        # Weight norm cap
        for a in range(self.num_actions):
            col_norm = np.linalg.norm(self.W[:, a])
            if col_norm > self.w_max_norm:
                self.W[:, a] *= self.w_max_norm / col_norm

        if done:
            self.e_theta[:] = 0.0
            self.e_W[:] = 0.0

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1

        return delta

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "approx": self.approx,
            "feature_size": self.feature_map.feature_size,
        }

    def save(self, path: str):
        """Save weights and meta-data to a numpy .npz file."""
        np.savez(
            path,
            theta=self.theta,
            W=self.W,
            epsilon=np.array([self.epsilon]),
            training_steps=np.array([self.training_steps]),
        )

    def load(self, path: str):
        """Load weights from a .npz file saved by save()."""
        data = np.load(path, allow_pickle=False)
        self.theta = data["theta"]
        self.W = data["W"]
        self.epsilon = float(data["epsilon"][0])
        self.training_steps = int(data["training_steps"][0])
        self.e_theta = np.zeros_like(self.theta)
        self.e_W = np.zeros_like(self.W)

    def reset_traces(self):
        """Zero out eligibility traces (call at the start of each episode)."""
        self.e_theta[:] = 0.0
        self.e_W[:] = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-Agent Manager
# ═══════════════════════════════════════════════════════════════════════════════


class MultiAgentActorCritic:
    """
    Manages a collection of independent ActorCriticAgent instances,
    one per traffic-light intersection (TLS) in the SUMO network.
    """

    def __init__(
        self,
        tls_ids: list,
        state_size: int = 9,
        num_actions: int = 2,
        approx: str = "rbf",
        n_rbf: int = 5,
        n_tilings: int = 8,
        tiles_per_dim: int = 4,
        state_low=None,
        state_high=None,
        alpha_critic: float = 0.1,
        alpha_actor: float = 0.1,
        gamma: float = 0.95,
        lambda_: float = 0.9,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        delta_clip: float = 10.0,
        trace_max_norm: float = 10.0,
        w_max_norm: float = 50.0,
    ):
        self.tls_ids = list(tls_ids)
        self.agents: dict[str, ActorCriticAgent] = {}

        for tid in self.tls_ids:
            self.agents[tid] = ActorCriticAgent(
                agent_id=tid,
                state_size=state_size,
                num_actions=num_actions,
                approx=approx,
                n_rbf=n_rbf,
                n_tilings=n_tilings,
                tiles_per_dim=tiles_per_dim,
                state_low=state_low,
                state_high=state_high,
                alpha_critic=alpha_critic,
                alpha_actor=alpha_actor,
                gamma=gamma,
                lambda_=lambda_,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                delta_clip=delta_clip,
                trace_max_norm=trace_max_norm,
                w_max_norm=w_max_norm,
            )

        self.name = f"Multi-AC ({len(self.tls_ids)} agents, {approx.upper()})"

    # ── Batched API ───────────────────────────────────────────────────────

    def select_actions(self, states: dict) -> dict:
        """
        Select actions for all agents.

        Parameters
        ----------
        states : Dict[tls_id, state]

        Returns
        -------
        actions : Dict[tls_id, int]
        """
        return {
            tid: self.agents[tid].select_action(states[tid])
            for tid in self.tls_ids
            if tid in states
        }

    def learn(
        self,
        states: dict,
        actions: dict,
        rewards: dict,
        next_states: dict,
        done: bool,
    ) -> dict:
        """
        Update all agents from their individual transitions.

        Returns
        -------
        td_errors : Dict[tls_id, float]
        """
        td_errors = {}
        for tid in self.tls_ids:
            if tid in states:
                td_errors[tid] = self.agents[tid].learn(
                    state=states[tid],
                    action=actions[tid],
                    reward=rewards[tid],
                    next_state=next_states[tid],
                    done=done,
                )
        return td_errors

    def reset_traces(self):
        """Reset eligibility traces for all agents (start of episode)."""
        for agent in self.agents.values():
            agent.reset_traces()

    def update_epsilon(self, episode: int, phase_boundaries: list):
        """Apply phase-based epsilon schedule to all agents."""
        for agent in self.agents.values():
            if episode < phase_boundaries[0]:
                pass  # epsilon decays naturally
            elif episode < phase_boundaries[1]:
                if agent.epsilon < 0.1:
                    agent.epsilon = 0.1
            else:
                agent.epsilon = 0.0

    # ── Per-agent stats ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return aggregate statistics across all agents."""
        epsilons = [a.epsilon for a in self.agents.values()]
        steps = [a.training_steps for a in self.agents.values()]
        return {
            "num_agents": len(self.agents),
            "avg_epsilon": float(np.mean(epsilons)),
            "min_epsilon": float(np.min(epsilons)),
            "total_training_steps": int(np.sum(steps)),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str):
        """Save all agent weights into *directory* as separate .npz files."""
        os.makedirs(directory, exist_ok=True)
        for tid, agent in self.agents.items():
            safe_name = tid.replace("/", "_").replace("\\", "_")
            path = os.path.join(directory, f"agent_{safe_name}.npz")
            agent.save(path)
        print(f"  {len(self.agents)} agents saved to {directory}")

    def load(self, directory: str):
        """Load agent weights from .npz files in *directory*."""
        loaded = 0
        for tid, agent in self.agents.items():
            safe_name = tid.replace("/", "_").replace("\\", "_")
            path = os.path.join(directory, f"agent_{safe_name}.npz")
            if os.path.exists(path):
                agent.load(path)
                loaded += 1
            else:
                print(f"  WARNING: No saved weights for agent '{tid}'")
        print(f"  {loaded}/{len(self.agents)} agents loaded from {directory}")
