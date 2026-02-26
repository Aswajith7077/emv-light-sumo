# Actor-Critic Method – How It Works

This document explains the Actor-Critic algorithm implemented in `src/actor_critic/`.

---

## The Core Idea

Actor-Critic is a **Reinforcement Learning (RL)** method that combines two separate ideas:

| Component | Role | Analogy |
|-----------|------|---------|
| **Critic** | Estimates *how good* the current state is (value function) | A coach watching and judging |
| **Actor** | Decides *which action to take* (policy) | A player executing moves |

The Critic tells the Actor "that was better / worse than expected", and the Actor adjusts its behaviour accordingly. Neither works well alone — the Critic improves the Actor's updates, and the Actor's actions generate experience for the Critic to learn from.

---

## The Two Learnable Components

### 1. The Critic — Value Function V(s)

The Critic maintains a weight vector **θ** and estimates the value of a state as a dot product:

```
V(s) = θᵀ · φ(s)
```

- `φ(s)` is a **feature vector** that represents the state (explained below).
- `θ` is learned over time so that `V(s)` matches the true expected return.

In code (`agent.py`):
```python
self.theta = np.zeros(fs, dtype=np.float64)   # Critic weights θ
v_s = float(self.theta @ phi)                  # V(s) = θᵀφ
```

### 2. The Actor — Policy π(a|s)

The Actor maintains a weight matrix **W** (one column per action) and uses a **softmax policy**:

```
h(s, a) = W[:, a]ᵀ · φ(s)      # preference score for action a
π(a|s)  = softmax(h(s, ·))      # probability distribution over actions
```

In code:
```python
self.W = np.zeros((fs, num_actions), dtype=np.float64)  # Actor weights W
h = phi @ self.W                                         # preferences
probs = softmax(h)                                       # policy π(a|s)
```

---

## The Learning Signal — TD Error (δ)

Every time a transition `(s, a, r, s')` is observed, the agent computes the **Temporal Difference (TD) error**:

```
δ = r + γ · V(s') - V(s)
```

| Term | Meaning |
|------|---------|
| `r` | Reward just received |
| `γ · V(s')` | Discounted estimate of future value |
| `V(s)` | What we already thought this state was worth |

- **δ > 0** → the outcome was *better than expected* → reinforce the action
- **δ < 0** → the outcome was *worse than expected* → discourage the action
- **δ = 0** → perfect prediction, no update needed

In code:
```python
delta = reward + self.gamma * v_s_next - v_s
```

---

## Eligibility Traces — Memory of the Past

Instead of updating only the most recent step, **eligibility traces** act as a short-term memory that spreads credit backward through recent states:

```
e_θ ← γλ · e_θ + φ(s)                       # Critic trace
e_W ← γλ · e_W + ∇log π(a|s)               # Actor trace
```

- `λ` (lambda) controls how far back credit flows. `λ=0` is pure one-step TD; `λ=1` resembles Monte Carlo.
- This project uses `λ = 0.9`, which means recent steps get near-full credit and older ones decay exponentially.

In code:
```python
self.e_theta = self.gamma * self.lambda_ * self.e_theta + phi
self.e_W     = self.gamma * self.lambda_ * self.e_W     + grad_log_pi
```

---

## The Weight Updates

Once we have `δ` and the traces, both components update their weights:

```
θ ← θ + α_c · δ · e_θ    # Critic: improve value estimate
W ← W + α_a · δ · e_W    # Actor: improve policy
```

- `α_c` and `α_a` are learning rates for the Critic and Actor respectively (both 0.1 here).
- The same TD error `δ` drives both updates — positive δ strengthens recent actor choices, negative δ weakens them.

In code:
```python
self.theta += self.alpha_critic * delta * self.e_theta
self.W     += self.alpha_actor  * delta * self.e_W
```

At episode end, traces are reset to zero:
```python
if done:
    self.e_theta[:] = 0.0
    self.e_W[:] = 0.0
```

---

## The Policy Gradient — How the Actor Learns

The Actor update uses the **score function** (log-derivative trick):

```
∇_W log π(a|s) = φ(s) · (1[a=a'] - π(a'|s))   for each action a'
```

This is the `grad_log_pi` term. It tells us:
- Increase the preference of the **chosen action** (`1[a=a']` = 1).
- Decrease the preference of **all other actions** proportionally to their probability.
- Scale the whole update by `δ` — so only reinforce when the outcome was *better than expected*.

In code:
```python
indicator = np.zeros(self.num_actions)
indicator[action] = 1.0
grad_log_pi = np.outer(phi, indicator - probs)   # (feature_size, num_actions)
```

---

## Feature Maps — Representing State

Raw state (e.g. queue lengths at 8 intersections + current phase) is mapped into a rich feature vector `φ(s)` using one of two methods:

### RBF (Radial Basis Functions) — `feature_maps.py`
Smooth, dense features. Places Gaussian "bumps" across the state space:
```
φ_i(s) = exp(-‖s_norm - μ_i‖² / (2σ²))
```
- `n_rbf` centers per dimension (paper tested 2, 5, 8 — **5 wins**).
- State is normalized to `[0,1]` before computing distances.
- A bias term (1.0) is appended.

### Tile Coding — `feature_maps.py`
Sparse, binary features. Divides the state space into overlapping grids (tilings):
- Each tiling is offset slightly from the others.
- For any state, exactly **one tile per tiling** is active → sparse `φ(s)` with `n_tilings` ones.
- Faster and more memory-efficient for large state spaces.

---

## Action Selection — ε-greedy over Softmax

```python
if random.random() < self.epsilon:
    return random.randint(0, self.num_actions - 1)  # explore randomly
else:
    return argmax(softmax(W.T @ phi))               # exploit policy
```

Exploration (`ε`) starts high and decays over training, following the 3-phase schedule from the paper:

| Phase | Episodes | Behaviour |
|-------|----------|-----------|
| 1 | 1–6  | High exploration (ε: 0.9 → 0.1) |
| 2 | 7–13 | Low exploration (ε clamped at 0.1) |
| 3 | 14–20| Pure exploitation (ε = 0.0) |

---

## Full Update Flow (Per Step)

```
1. Observe state s
2. Compute φ(s) via feature map
3. Select action a ~ π(·|s)  [or random with prob ε]
4. Execute a, receive reward r, observe s'
5. Compute φ(s')
6. Compute TD error:  δ = r + γ·V(s') - V(s)
7. Compute grad_log_pi = φ(s) ⊗ (1[a] - π(·|s))
8. Update traces:
      e_θ ← γλ·e_θ + φ(s)
      e_W ← γλ·e_W + grad_log_pi
9. Update weights:
      θ ← θ + α_c · δ · e_θ
      W ← W + α_a · δ · e_W
10. If episode done, reset traces to 0
11. Decay ε
```

---

## Why Actor-Critic Over Pure Methods?

| Method | Problem |
|--------|---------|
| **Pure Policy Gradient** (REINFORCE) | High variance — uses full episode return, very noisy updates |
| **Pure Value (Q-Learning / DQN)** | No direct policy gradient — needs argmax, struggles with continuous actions |
| **Actor-Critic** | Low variance (Critic acts as baseline), direct policy optimization, works online (no replay needed) |

The Critic's value estimate replaces the noisy Monte Carlo return, dramatically reducing variance while keeping the Actor's direct gradient signal.

---

## Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `alpha_critic` (α_c) | 0.1 | Critic learning rate |
| `alpha_actor` (α_a) | 0.1 | Actor learning rate |
| `gamma` (γ) | 0.95 | Discount factor — how much future rewards matter |
| `lambda_` (λ) | 0.9 | Eligibility trace decay — credit assignment range |
| `n_rbf` | 5 | RBF centers per dimension (paper-tuned) |
| `epsilon_start` | 0.9 | Initial exploration rate |
| `epsilon_decay` | 0.9991 | Per-step decay (~0.9→0.1 over 30,000 steps) |
