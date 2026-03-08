    
```python

"""
    Continuous Actor-Critic agent with linear function approximation.

    Supports two feature mapping methods:
        - 'rbf'  : Radial Basis Functions (Gaussian, smooth)
        - 'tile' : Tile Coding (sparse, piecewise constant)

    Parameters
    ----------
    state_size : int
        Dimensionality of the raw state (e.g. 9 for queue lengths + phase).
    num_actions : int
        Number of discrete actions (e.g. 2: keep / switch phase).
    approx : str
        Feature approximation method: 'rbf' (default) or 'tile'.
    n_rbf : int
        RBF centres per dimension (used when approx='rbf'). Paper: 2/5/8.
    n_tilings : int
        Number of tile-coding tilings (used when approx='tile').
    tiles_per_dim : int
        Tiles per dimension per tiling (used when approx='tile').
    state_low : array-like or None
        Lower bound per state dimension for normalisation.
    state_high : array-like or None
        Upper bound per state dimension for normalisation.
    alpha_critic : float
        Critic (value) learning rate α_c. Paper default: 0.1.
    alpha_actor : float
        Actor (policy) learning rate α_a. Paper default: 0.1.
    gamma : float
        Discount factor γ.
    lambda_ : float
        Eligibility trace decay λ. Paper default: 0.9.
    epsilon_start : float
        Initial exploration rate ε.
    epsilon_end : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiplicative decay applied per learn() call.

"""

"""
Actor-Critic Agent for Adaptive Traffic Signal Control (A-CAT).

Implements the continuous Actor-Critic with linear function approximation
and eligibility traces, as described in:

  "Adaptive traffic signal control with actor-critic methods in a real-world
   traffic network with different traffic disruption events"
  Transportation Research Part C, 2017

Update equations (one-step TD, linear FA):
    φ     = feature_map(s)
    φ'    = feature_map(s')
    δ     = r + γ·θᵀφ'  − θᵀφ         (TD error)
    e_θ  ← γλ·e_θ + φ                 (critic eligibility trace)
    e_W  ← γλ·e_W + φ(∇ log π(a|s))  (actor eligibility trace)
    θ    ← θ + α_c·δ·e_θ
    W    ← W + α_a·δ·e_W

Policy (softmax):
    h(s, a) = W[:, a]ᵀ · φ(s)
    π(a|s)  = softmax(h(s, ·))
"""


```