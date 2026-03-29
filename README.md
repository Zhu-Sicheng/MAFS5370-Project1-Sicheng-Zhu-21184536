# MAFS5370 Project 1 — Sicheng Zhu (21184536)

This repository contains a single Jupyter notebook implementing and benchmarking multiple reinforcement learning (RL) algorithms on a discrete-time portfolio allocation problem.

## 1. Project Overview
The task is a finite-horizon portfolio rebalancing problem. At each time step, the agent chooses how to adjust portfolio weights (subject to constraints). The market then realizes stochastic returns, portfolio wealth evolves, and the agent receives a terminal utility reward based on final wealth.

The objective is to learn a policy that maximizes expected CRRA-style utility of terminal wealth over a fixed horizon.

## 2. Environment (PortfolioEnv)
The environment is a toy market with 3 risky assets and 1 risk-free asset (cash). Returns are generated i.i.d. each step.

### 3.1 Market Dynamics
- **Horizon**: `T = 8` steps.
- **Risky assets**: `n_assets = 3`.
  - Expected returns: `a = [-0.5, 0.5, 0.0]`
  - Variances: `s = [0.25, 0.25, 0.5]` (so std = `sqrt(s)`)
  - Realized risky returns per step: `stock_returns ~ Normal(a, sqrt(s))` (independent across assets and time)
- **Risk-free asset**: constant return `r = 0.02`.

### 3.2 Portfolio Update
- **Weights**: `weights = [w_cash, w_1, w_2, w_3]` with sum-to-one enforced.
- **Action**: the agent chooses deltas for risky weights only: `deltas = [Δw_1, Δw_2, Δw_3]`.
  - After applying deltas, cash weight is set as the residual:
    - `w_cash = 1 - (w_1 + w_2 + w_3)`
- **Portfolio return** (one-step):
  - `R_p = Σ_i (w_i * r_i) + w_cash * r`
- **Wealth evolution**:
  - `W_{t+1} = W_t * (1 + R_p)`

### 3.3 Constraints & Feasibility
Constraints are enforced inside `_check_feasible(deltas)`:
- Total turnover limit per step: `Σ_i |Δw_i| ≤ 0.1`
- No short selling by default: all weights must be non-negative when `ALLOW_SHORT = False`
  - Note: short selling may be optimal in this toy setup, but experiments in the notebook set `ALLOW_SHORT = False`.

If an action is infeasible:
- The environment advances time by 1 anyway.
- A small penalty is given at intermediate steps (`reward = -0.01`), while the terminal step still returns terminal utility.

### 3.4 State Representation
The environment state is discretized to keep tabular learning feasible:
- `time_norm = time / T`
- `wealth_discrete = round(current_wealth, 2)`
- `weights_discrete = tuple(round(weights, 3))`
- State tuple:
  - `(time_norm, wealth_discrete, w_cash, w_1, w_2, w_3)`

### 3.5 Reward (Terminal Utility)
Reward is only given at terminal time:
- `U(W) = -exp(-γ W)` with `γ = 1.0`
- For non-terminal feasible steps, reward is `0`.

## 3. Action Space (Discrete)
The notebook discretizes each risky-asset weight change into 21 values:
- `{ -0.10, -0.09, ..., -0.01, 0, 0.01, ..., 0.10 }`

Then it enumerates all 3D delta vectors satisfying `Σ|Δw_i| ≤ 0.1`. With `n_assets = 3`, this produces:
- **1537** discrete actions in total (printed by the notebook).

## 4. Algorithms Implemented
All algorithms interact with the same `PortfolioEnv` and share the same discrete action set.

### 5.1 Greedy (Baseline)
- Enumerates all feasible actions each step and chooses the action maximizing **one-step expected return**:
  - `E[R_p] = Σ w_i * a_i + w_cash * r`

### 5.2 Q-Learning (Tabular, Adaptive Step Size)
- Uses a tabular Q-function `Q(state, action)` with ε-greedy exploration (`ε = 0.2`).
- Discount `γ = 0.99`.
- Uses an Adam-style adaptive learning rate update on TD error (implemented via per-(state,action) `m` and `v` accumulators).

### 5.3 REINFORCE (Policy Gradient)
- Policy network: MLP with two 128-unit ReLU layers and Softmax output over 1537 discrete actions.
- Updates policy via Monte Carlo returns (discount `γ = 0.99`) and standardized returns for variance reduction.

### 5.4 Actor-Critic (A2C-style, Single Trajectory)
- Actor network: same MLP policy as REINFORCE.
- Critic network: MLP value function.
- Uses advantage `A_t = G_t - V(s_t)` and optimizes:
  - Actor loss + value MSE + entropy regularization
- Applies gradient clipping to stabilize learning.

### 5.5 PPO (Clipped Objective)
- Actor + critic networks.
- Uses GAE with `λ = 0.95`.
- PPO clip `ε = 0.2`, `update_epochs = 4`, `batch_size = 64`.

### 5.6 A3C (Multi-Worker Trajectory Collection)
- Collects multiple trajectories per “episode” using `n_workers = 4`.
- Updates shared (global) actor/critic parameters using aggregated losses.
  - Implementation is multi-trajectory but not truly asynchronous multiprocessing.

### 5.7 TRPO (Trust-Region Style Update)
- Policy network + value network.
- Approximates a trust-region update using:
  - KL divergence constraint
  - Conjugate gradient solver for the natural-gradient direction
  - Value-function regression steps

## 5. Training & Evaluation Protocol
The notebook runs the following pipeline for each algorithm:
- **Training episodes**: `episodes = 3000`
- **Evaluation episodes**: `eval_episodes = 500`
- **Evaluation policy**: greedy w.r.t. learned policy outputs
  - Greedy baseline: uses its own one-step maximization rule
  - Q-learning: chooses `argmax_a Q(s,a)`
  - Neural policies: chooses `argmax_a π(a|s)` (no sampling)

It logs wealth/utility histories and then plots:
- Moving-average learning curves (utility & wealth)
- Boxplots of evaluation utility
- Bar charts for mean utility and mean wealth (with std error bars)

## 6. Results (From Notebook Output)
After training (3000 episodes) and evaluation (500 episodes), the notebook reports:

| Algorithm | Mean Utility | Std Utility | Mean Wealth | Std Wealth |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy** | **-0.0815** | 0.1450 | **5.2517** | 4.4217 |
| **A3C** | -0.1303 | 0.1751 | 3.8964 | 3.4623 |
| **REINFORCE** | -0.1471 | 0.1580 | 2.8295 | 1.8696 |
| **Actor-Critic** | -0.2567 | 0.1549 | 1.5729 | 0.7091 |
| **PPO** | -0.2584 | 0.1067 | 1.4548 | 0.4944 |
| **Q-Learning** | -0.3276 | 0.1142 | 1.1777 | 0.3585 |
| **TRPO** | -0.6464 | 0.2123 | 0.5089 | 0.4245 |

## 7. Figures
<img width="1590" height="989" alt="image" src="https://github.com/user-attachments/assets/3c143872-97fe-41a8-aa39-8dceb1c0607e" />


## 8. Reproducibility Notes
- Random seeds are fixed at the top of the notebook:
  - NumPy: 42
  - Python `random`: 42
  - PyTorch: 42

## 9. Usage
### 9.1 Requirements
- Python 3.12 (the notebook metadata uses 3.12.7)
- Packages:
  - `numpy`
  - `matplotlib`
  - `torch`
  - `tqdm`

### 9.2 Run
1. Open `Project1-MAFS5370.ipynb` in Jupyter.
2. Run all cells from top to bottom.
3. The notebook prints training/evaluation summaries and saves two PNG figures in the project folder.

---
MAFS5370 Project 1 Sicheng Zhu 21184536
