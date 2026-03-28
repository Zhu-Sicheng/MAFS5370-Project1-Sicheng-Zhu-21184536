# MAFS5370 Project 1: Reinforcement Learning for Dynamic Portfolio Optimization

## 1. Project Overview
This project explores the application of various Reinforcement Learning (RL) algorithms to a discrete-time portfolio management problem. The objective is to determine an optimal asset allocation strategy that maximizes the cumulative CRRA utility of final wealth over a fixed time horizon.

We implement and compare seven different algorithms, ranging from simple baselines to sophisticated actor-critic methods, to evaluate their performance in a simulated financial environment.

## 2. Environment Description
The environment simulates a market with both risky and risk-free assets:
- **Time Horizon (T)**: 8 investment periods.
- **Asset Universe**: 
    - 3 Risky Assets with varying expected returns `[-0.5, 0.5, 0.0]` and variances `[0.25, 0.25, 0.5]`.
    - 1 Risk-free Asset with a constant return of `0.02`.
- **Initial State**: Wealth $W_0 = 1.0$, with an equal weight distribution `[0.25, 0.25, 0.25, 0.25]`.
- **Constraints**: 
    - No short selling is allowed.
    - Weights must sum to 1.
    - Transaction limits: Maximum weight adjustment per asset is $\pm 0.1$ per step, with a total absolute adjustment limit of $0.1$.
- **Utility Function**: Constant Relative Risk Aversion (CRRA) utility:
  $$U(W) = -\exp(-\gamma W)$$
  where the risk aversion parameter $\gamma = 1.0$.

## 3. Algorithms Implemented
The following algorithms are evaluated:
1.  **Greedy (Baseline)**: Always selects the action that maximizes the one-step expected return.
2.  **Q-Learning**: A value-based method using an Adam-style adaptive learning rate for efficient convergence.
3.  **REINFORCE**: A basic policy gradient method that updates the policy network based on the total return of an episode.
4.  **Actor-Critic**: Combines value-based and policy-based methods to reduce variance in gradient estimates.
5.  **A3C (Asynchronous Advantage Actor-Critic)**: Uses multiple parallel workers to stabilize training.
6.  **PPO (Proximal Policy Optimization)**: A robust policy gradient method that uses a clipped objective function to prevent large updates.
7.  **TRPO (Trust Region Policy Optimization)**: Constrains policy updates within a trust region defined by KL-divergence.

## 4. Performance Results
After training for 3000 episodes and evaluating over 500 test episodes, the performance metrics for each algorithm are summarized below:

| Algorithm | Mean Utility | Std Utility | Mean Wealth | Std Wealth |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy** | **-0.0815** | 0.1450 | **5.2517** | 4.4217 |
| **Q-Learning** | -0.3276 | 0.1142 | 1.1777 | 0.3585 |
| **REINFORCE** | -0.1471 | 0.1580 | 2.8295 | 1.8696 |
| **Actor-Critic** | -0.2567 | 0.1549 | 1.5729 | 0.7091 |
| **A3C** | -0.1303 | 0.1751 | 3.8964 | 3.4623 |
| **PPO** | -0.2584 | 0.1067 | 1.4548 | 0.4944 |
| **TRPO** | -0.6464 | 0.2123 | 0.5089 | 0.4245 |

### Result Analysis
- **Greedy Baseline**: Surprisingly, the Greedy algorithm achieved the highest mean wealth and utility. This is likely due to the specific return distribution (one asset has a very high positive return of 0.5), allowing a simple "chase the return" strategy to perform well.
- **A3C & REINFORCE**: These algorithms performed the best among the RL agents, successfully capturing a significant portion of the available market growth while maintaining reasonable utility.
- **Stability**: Methods like **Q-Learning** and **PPO** showed lower standard deviation in wealth, indicating more stable (though perhaps more conservative) strategies.
- **TRPO Performance**: In this specific discrete and low-horizon environment, TRPO was overly conservative, leading to the lowest wealth accumulation.

## 5. Visualizations
The project generates comprehensive plots comparing the learning curves (Utility and Wealth) and final performance distributions across all algorithms. The final summary plot is saved as `portfolio_rl_comparison.png`.

## 6. Usage
To run the analysis:
1. Ensure you have the required dependencies: `numpy`, `matplotlib`, `torch`, `tqdm`.
2. Open and execute the Jupyter Notebook `Project1-MAFS5370.ipynb`.
3. The results will be displayed inline and the comparison chart will be exported.

---
*MAFS5370 Project 1 - Reinforcement Learning in Finance*
