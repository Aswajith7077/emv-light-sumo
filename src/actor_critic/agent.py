import random

import numpy as np

from .feature_maps import RBFFeatureMap, TileCodingFeatureMap


class ActorCriticAgent:
    def __init__(
        self,
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

        if approx == "rbf":
            self.name = f"Actor-Critic (RBF-{n_rbf})"
            self.feature_map = RBFFeatureMap(
                state_size=state_size,
                n_rbf=n_rbf,
                state_low=state_low,
                state_high=state_high,
            )
        elif approx == "tile":
            self.name = f"Actor-Critic (Tile-{n_tilings}x{tiles_per_dim})"
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

        self.theta = np.zeros(fs, dtype=np.float64)

        self.W = np.zeros((fs, num_actions), dtype=np.float64)

        self.e_theta = np.zeros(fs, dtype=np.float64)
        self.e_W = np.zeros((fs, num_actions), dtype=np.float64)

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

        self.e_theta = self.gamma * self.lambda_ * self.e_theta + phi
        self.e_W = self.gamma * self.lambda_ * self.e_W + grad_log_pi

        e_theta_norm = np.linalg.norm(self.e_theta)
        if e_theta_norm > self.trace_max_norm:
            self.e_theta *= self.trace_max_norm / e_theta_norm

        e_W_norm = np.linalg.norm(self.e_W)
        if e_W_norm > self.trace_max_norm:
            self.e_W *= self.trace_max_norm / e_W_norm

        self.theta += self.alpha_critic * delta * self.e_theta
        self.W += self.alpha_actor * delta * self.e_W

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

    def get_stats(self) -> dict:
        return {
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
        print(f"  Actor-Critic agent saved to {path}")

    def load(self, path: str):
        """Load weights from a .npz file saved by save()."""
        data = np.load(path, allow_pickle=False)
        self.theta = data["theta"]
        self.W = data["W"]
        self.epsilon = float(data["epsilon"][0])
        self.training_steps = int(data["training_steps"][0])
        self.e_theta = np.zeros_like(self.theta)
        self.e_W = np.zeros_like(self.W)
        print(f"  Actor-Critic agent loaded from {path}")
        print(f"  Training steps: {self.training_steps}, epsilon: {self.epsilon:.4f}")
