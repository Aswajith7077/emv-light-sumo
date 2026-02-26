"""
Feature mapping methods for continuous Actor-Critic (A-CAT).

Implements two function approximation methods from:
  "Adaptive traffic signal control with actor-critic methods in a real-world
   traffic network with different traffic disruption events"
  Transportation Research Part C, 2017

Both classes expose:
    __call__(state)  -> np.ndarray  (feature vector φ(s), includes bias)
    feature_size     -> int
"""

import numpy as np


class RBFFeatureMap:
    """
    Radial Basis Function (Gaussian) feature map.

    Creates a grid of n_rbf^state_size RBF centers and computes:
        φ_i(s) = exp(-‖s_norm - μ_i‖² / (2σ²))

    The state is first normalised to [0, 1] using provided bounds.
    A constant bias feature (1.0) is appended.

    Parameters
    ----------
    state_size : int
        Dimensionality of the raw state vector.
    n_rbf : int
        Number of RBF centers per dimension (paper tests 2, 5, 8).
    state_low : array-like, optional
        Per-dimension lower bounds for normalisation. Defaults to 0.
    state_high : array-like, optional
        Per-dimension upper bounds for normalisation. Defaults to 1.
    sigma : float, optional
        Width of each Gaussian. Defaults to 1 / (n_rbf - 1) or 0.5 for n_rbf=1.
    """

    def __init__(
        self,
        state_size: int,
        n_rbf: int = 5,
        state_low=None,
        state_high=None,
        sigma: float = None,
    ):
        self.state_size = state_size
        self.n_rbf = n_rbf

        # Normalisation bounds
        self.state_low = (
            np.zeros(state_size)
            if state_low is None
            else np.array(state_low, dtype=np.float64)
        )
        self.state_high = (
            np.ones(state_size)
            if state_high is None
            else np.array(state_high, dtype=np.float64)
        )

        # Build center grid using meshgrid over [0,1]^state_size
        centers_per_dim = np.linspace(0.0, 1.0, n_rbf)  # shape (n_rbf,)
        grids = np.meshgrid(*[centers_per_dim] * state_size, indexing="ij")
        # Stack into (n_centers, state_size)
        self._centers = np.column_stack([g.ravel() for g in grids])  # (n_rbf^d, d)

        # Sigma
        if sigma is not None:
            self._sigma2 = sigma**2
        else:
            spread = 1.0 / (n_rbf - 1) if n_rbf > 1 else 0.5
            self._sigma2 = spread**2

        self._n_centers = self._centers.shape[0]
        # +1 for bias
        self._feature_size = self._n_centers + 1

    @property
    def feature_size(self) -> int:
        return self._feature_size

    def _normalise(self, state: np.ndarray) -> np.ndarray:
        denom = self.state_high - self.state_low
        denom = np.where(denom == 0, 1.0, denom)
        return (state - self.state_low) / denom

    def __call__(self, state) -> np.ndarray:
        """
        Compute RBF feature vector φ(s).

        Parameters
        ----------
        state : array-like, shape (state_size,)

        Returns
        -------
        phi : np.ndarray, shape (feature_size,)
        """
        s = np.array(state, dtype=np.float64).flatten()[: self.state_size]
        s_norm = self._normalise(s)

        diff = s_norm[np.newaxis, :] - self._centers  # (n_centers, d)
        dist2 = np.sum(diff**2, axis=1)  # (n_centers,)
        phi = np.exp(-dist2 / (2.0 * self._sigma2))  # (n_centers,)

        return np.append(phi, 1.0)  # bias


class TileCodingFeatureMap:
    """
    Tile Coding feature map (grid-based, sparse binary).

    Partitions the (normalised) state space using multiple offset tilings.
    Each tiling contributes exactly one active tile per state → sparse φ(s).
    A constant bias feature (1.0) is appended.

    Parameters
    ----------
    state_size : int
        Dimensionality of the raw state.
    n_tilings : int
        Number of overlapping tilings (paper uses values like 4, 8).
    tiles_per_dim : int
        Number of tiles per dimension per tiling.
    state_low : array-like, optional
        Per-dimension lower bounds. Defaults to 0.
    state_high : array-like, optional
        Per-dimension upper bounds. Defaults to 1.
    """

    def __init__(
        self,
        state_size: int,
        n_tilings: int = 8,
        tiles_per_dim: int = 4,
        state_low=None,
        state_high=None,
    ):
        self.state_size = state_size
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim

        self.state_low = (
            np.zeros(state_size)
            if state_low is None
            else np.array(state_low, dtype=np.float64)
        )
        self.state_high = (
            np.ones(state_size)
            if state_high is None
            else np.array(state_high, dtype=np.float64)
        )

        # Total tiles per tiling: tiles_per_dim^state_size
        self._tiles_per_tiling = tiles_per_dim**state_size
        # +1 for bias
        self._feature_size = n_tilings * self._tiles_per_tiling + 1

        # Pre-compute tile widths and offsets for each tiling
        self._tile_widths = 1.0 / tiles_per_dim  # normalised
        # Each tiling is offset by a fraction of tile width
        self._offsets = np.array(
            [i / n_tilings * self._tile_widths for i in range(n_tilings)]
        )  # shape (n_tilings,)

    @property
    def feature_size(self) -> int:
        return self._feature_size

    def _normalise(self, state: np.ndarray) -> np.ndarray:
        denom = self.state_high - self.state_low
        denom = np.where(denom == 0, 1.0, denom)
        return np.clip((state - self.state_low) / denom, 0.0, 1.0 - 1e-9)

    def __call__(self, state) -> np.ndarray:
        """
        Compute sparse tile-coding feature vector φ(s).

        Parameters
        ----------
        state : array-like, shape (state_size,)

        Returns
        -------
        phi : np.ndarray, shape (feature_size,)  (sparse binary + bias)
        """
        s = np.array(state, dtype=np.float64).flatten()[: self.state_size]
        s_norm = self._normalise(s)

        phi = np.zeros(self._feature_size, dtype=np.float64)

        for t in range(self.n_tilings):
            # Shift state by tiling offset
            shifted = s_norm + self._offsets[t]
            # Compute tile index per dimension
            tile_idx_per_dim = np.floor(shifted / self._tile_widths).astype(int)
            tile_idx_per_dim = np.clip(tile_idx_per_dim, 0, self.tiles_per_dim - 1)

            # Encode multi-dim index into flat index (row-major)
            flat_idx = int(
                np.ravel_multi_index(
                    tile_idx_per_dim, [self.tiles_per_dim] * self.state_size
                )
            )

            # Offset by tiling
            feature_idx = t * self._tiles_per_tiling + flat_idx
            phi[feature_idx] = 1.0

        phi[-1] = 1.0  # bias
        return phi
