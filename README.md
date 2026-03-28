# Portfolio Optimization with Reinforcement Learning
## MAFS5370 Project 1 Sicheng Zhu 21184536

---

## 1. Problem Formulation

### 1.1 Investment Problem
Investor allocates wealth among n risky assets and a risk-free asset over T periods. At each step, portfolio is rebalanced by adjusting weights.

**Assets:**
- Risk-free: guaranteed return r
- Risky assets i: returns R_i ~ N(a_i, s_i)

### 1.2 State Representation
state = (time_norm, wealth_discrete, w1, w2, ..., wn)
- `time_norm`: Normalized time step (t/T)
- `wealth_discrete`: Current wealth rounded to 2 decimals
- `w_i`: Portfolio weights rounded to 3 decimals

### 1.3 Action Space
- Weight changes Δw_i ∈ [-0.10, 0.10]
- Σ|Δw_i| ≤ 0.10
- Total actions: 4,370 (for n_assets=3)

### 1.4 Reward Structure
- Terminal reward: `U(W) = -exp(-γ * W)` (CRRA utility)
- Intermediate rewards: 0 for feasible actions, -0.01 for infeasible actions

### 1.5 Objective
Maximize expected terminal utility:
max E[U(W_T)] = max E[-exp(-γ * W_T)]


---

## 2. Environment Class: PortfolioEnv

### Parameters
| Parameter | Description |
|-----------|-------------|
| `T` | Investment horizon |
| `n_assets` | Number of risky assets |
| `a_list` | Expected returns for risky assets |
| `s_list` | Variances of risky asset returns |
| `r` | Risk-free rate |
| `p_list` | Initial portfolio weights |
| `risk_aversion` | Risk aversion parameter γ |
| `allow_short` | Whether short selling is permitted |

### Key Methods
- `reset()`: Initialize environment
- `step(deltas)`: Execute action, return (next_state, reward, done)
- `_check_feasible(deltas)`: Validate action
- `_utility(wealth)`: Compute CRRA utility

---

## 3. Implemented Algorithms

### 3.1 GreedyAgent
Deterministic baseline selecting action maximizing expected one-period return.

### 3.2 AdaptiveQLearningAgent
Tabular Q-learning with adaptive learning rate strategies:
- `fixed`: Constant learning rate
- `step_decay`: Exponential decay every 100 episodes
- `td_error`: Adjusts based on TD-error trends
- `adam`: Per-state-action Adam optimization

**Parameters:** α=0.1, γ=0.95, ε=0.2, min_α=0.001, max_α=0.5

### 3.3 REINFORCEAgent
Monte Carlo policy gradient with return normalization.

**Network:** `Linear(state_dim,128) → ReLU → Linear(128,128) → ReLU → Linear(128,n_actions) → Softmax`

**Parameters:** lr=0.001, γ=0.99

### 3.4 ActorCriticAgent
Combines policy network (actor) and value network (critic).

**Loss:** `actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss`

**Parameters:** lr=0.001, γ=0.99

### 3.5 A3CAgent
Asynchronous Actor-Critic with parallel workers.

**Parameters:** lr=0.001, γ=0.99, n_workers=4

### 3.6 PPOAgent
Proximal Policy Optimization with clipped surrogate objective.

**Parameters:** lr=0.0003, γ=0.99, ε=0.2, GAE λ=0.95, update_epochs=4, batch_size=64

### 3.7 TRPOAgent
Trust Region Policy Optimization with KL divergence constraint.

**Parameters:** γ=0.99, max_kl=0.01, GAE λ=0.95

---

## 4. Experimental Setup

| Parameter | Value |
|-----------|-------|
| T | 8 |
| n_assets | 3 |
| a | [-0.5, 0.5, 0.0] |
| s | [0.25, 0.25, 0.5] |
| r | 0.02 |
| Initial weights | [0.25, 0.25, 0.25, 0.25] |
| risk_aversion | 1.0 |
| allow_short | False |
| episodes | 3000 |
| eval_episodes | 500 |

---

## 5. Functions

### train_algorithm()
Trains agent for specified episodes with progress bar and logging.

### evaluate_algorithm()
Evaluates trained policy over multiple episodes, returns statistics:
- mean_wealth, std_wealth
- mean_utility, std_utility

### plot_comparison()
Generates 6 subplots:
1. Utility learning curves
2. Wealth learning curves
3. Utility distribution boxplots
4. Mean utility comparison
5. Mean wealth comparison
6. Normalized learning speed

---

## 6. Dependencies

```bash
numpy
torch
matplotlib
tqdm
```
---

## 7. Code Structure
```
├── PortfolioEnv
├── AdaptiveQLearningAgent
├── GreedyAgent
├── REINFORCEAgent
├── ActorCriticAgent
├── A3CAgent
├── PPOAgent
├── TRPOAgent
├── generate_actions()
├── train_algorithm()
├── evaluate_algorithm()
└── plot_comparison()
```
