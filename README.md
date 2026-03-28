# MAFS5370 Project 1: Reinforcement Learning for Portfolio Optimization

## 1. Project Introduction
This project implements and compares seven different Reinforcement Learning (RL) algorithms to solve a dynamic portfolio optimization problem. The goal is to maximize the expected CRRA utility of final wealth over a fixed time horizon by dynamically adjusting asset weights.

## 2. Environment Configuration
- **Time Horizon (T)**: 8 periods
- **Assets**: 3 Risky Assets + 1 Risk-free Asset
- **Parameters**:
  - Expected Returns: `[-0.5, 0.5, 0.0]`
  - Variances: `[0.25, 0.25, 0.5]`
  - Risk-free Rate: `0.02`
  - Initial Wealth: `1.0` (Equally weighted at start)
- **Constraints**: 
  - No short selling allowed.
  - Action space discretized: weight changes $\in [-0.1, 0.1]$ with step 0.01.
  - Total absolute weight adjustment per step $\leq 0.1$.
- **Utility Function**: CRRA Utility $U(W) = -\exp(-\gamma W)$ with $\gamma=1.0$.

## 3. Algorithms Evaluated
1.  **Greedy (Baseline)**: One-step return maximization.
2.  **Q-Learning**: Value-based RL with Adam adaptive learning rate.
3.  **REINFORCE**: Policy gradient method.
4.  **Actor-Critic**: Combined value and policy optimization.
5.  **A3C**: Asynchronous Advantage Actor-Critic.
6.  **PPO**: Proximal Policy Optimization (clipped objective).
7.  **TRPO**: Trust Region Policy Optimization (KL-divergence constraint).

## 4. Experimental Results
Based on the latest training run (3000 episodes) and evaluation (500 episodes), the performance metrics are as follows:

| Algorithm | Mean Utility | Std Utility | Mean Wealth | Std Wealth |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy** | **-0.0815** | 0.1450 | **5.2517** | 4.4217 |
| **A3C** | -0.1303 | 0.1751 | 3.8964 | 3.4623 |
| **REINFORCE** | -0.1471 | 0.1580 | 2.8295 | 1.8696 |
| **Actor-Critic** | -0.2567 | 0.1549 | 1.5729 | 0.7091 |
| **PPO** | -0.2584 | 0.1067 | 1.4548 | 0.4944 |
| **Q-Learning** | -0.3276 | 0.1142 | 1.1777 | 0.3585 |
| **TRPO** | -0.6464 | 0.2123 | 0.5089 | 0.4245 |

### Result Analysis
- **Dominance of Greedy Baseline**: The Greedy algorithm achieved the best results. This is due to the environment setup where one asset has a high expected return (0.5), and the short horizon (T=8) allows simple momentum-like strategies to outperform complex RL agents.
- **Top RL Performers**: **A3C** and **REINFORCE** are the strongest RL algorithms in this scenario, effectively learning to capture high-return assets while managing risk.
- **Stability vs. Performance**: **PPO** and **Q-Learning** demonstrated significantly lower variance (Std Wealth), indicating more stable and predictable portfolio management, albeit with lower average returns compared to A3C.
- **TRPO Challenges**: TRPO showed the lowest performance, likely due to its conservative trust-region updates which struggled to explore the discretized action space effectively within 3000 episodes.

## 5. Usage Instructions
1. Install dependencies: `numpy`, `torch`, `matplotlib`, `tqdm`.
2. Run the notebook `Project1-MAFS5370.ipynb`.
3. The training logs and comparison plots (`portfolio_rl_comparison.png`) will be generated automatically.

---
*MAFS5370 Project 1 - Academic Report*
