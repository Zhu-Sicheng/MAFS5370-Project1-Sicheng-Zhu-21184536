# MAFS5370 Project 1 Sicheng Zhu 21184536

## 1. Project Overview
This project explores the application of various Reinforcement Learning (RL) algorithms to a discrete-time portfolio management problem. The objective is to determine an optimal asset allocation strategy that maximizes the cumulative CRRA utility of final wealth over a fixed time horizon.

Seven algorithms are implemented and compared, ranging from simple baselines to sophisticated actor-critic methods, to evaluate their performance in a simulated financial environment.

## 2. Demo Environment Description
The environment simulates a market with both risky and risk-free assets:
- **Time Horizon (T)**: 8 investment periods.
- **Asset Universe**: 
    - 3 Risky Assets with varying expected returns `[-0.5, 0.5, 0.0]` and variances `[0.25, 0.25, 0.5]`.
    - 1 Risk-free Asset with a constant return of `0.02`.
- **Initial State**: Wealth $W_0 = 1.0$, with an equal weight distribution `[0.25, 0.25, 0.25, 0.25]`.
- **Constraints**: 
    - No short selling is allowed(but short selling is optimal).
    - Weights must sum to 1.
    - Transaction limits: Maximum weight adjustment per asset is $\pm 0.1$ per step, with a total absolute adjustment limit of $0.1$.
- **Utility Function**: Constant Relative Risk Aversion (CRRA) utility:
  $$U(W) = -\exp(-\gamma W)$$
  where the risk aversion parameter $\gamma = 1.0$.

## 3. Algorithms Implemented
The following algorithms are evaluated:
1.  **Greedy (Baseline)**: Always selects the action that maximizes the one-step expected return.
2.  **Q-Learning**: A value-based method using an Adam-style adaptive learning rate for efficient convergence.
3.  **REINFORCE**: A basic policy gradient method that updates the policy network based on the total return.
4.  **Actor-Critic**: Combines value-based and policy-based methods to reduce variance.
5.  **A3C (Asynchronous Advantage Actor-Critic)**: Uses parallel workers to stabilize training.
6.  **PPO (Proximal Policy Optimization)**: A robust policy gradient method using clipped updates.
7.  **TRPO (Trust Region Policy Optimization)**: Constrains policy updates within a trust region.

## 4. Key Findings
- **Experimental Results**: After 3000 episodes of training, the performance metrics (evaluated over 500 episodes) are summarized as follows:

| Algorithm | Mean Utility | Std Utility | Mean Wealth | Std Wealth |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy** | **-0.0815** | 0.1450 | **5.2517** | 4.4217 |
| **A3C** | -0.1303 | 0.1751 | 3.8964 | 3.4623 |
| **REINFORCE** | -0.1471 | 0.1580 | 2.8295 | 1.8696 |
| **Actor-Critic** | -0.2567 | 0.1549 | 1.5729 | 0.7091 |
| **PPO** | -0.2584 | 0.1067 | 1.4548 | 0.4944 |
| **Q-Learning** | -0.3276 | 0.1142 | 1.1777 | 0.3585 |
| **TRPO** | -0.6464 | 0.2123 | 0.5089 | 0.4245 |

- **Performance Analysis**:
    - **Greedy Dominance**: The Greedy algorithm performs best due to the high return of a specific asset (0.5) and horizon (T=8).
    - **RL Leaders**: **A3C** and **REINFORCE** are the top-performing RL agents, successfully learning growth strategies.
    - **Stability**: **PPO** and **Q-Learning** show lower variance in wealth, indicating more conservative but stable behavior.
    - **Visualizations**: The project generates plots comparing learning curves and distributions.

<img width="1590" height="989" alt="image" src="https://github.com/user-attachments/assets/b87fcc3a-66dd-4138-88f6-f2c38416a505" />

## 5. Usage
1. Ensure dependencies are installed: `numpy`, `matplotlib`, `torch`, `tqdm`.
2. Open and execute the Jupyter Notebook `Project1-MAFS5370.ipynb`.
3. Results and comparison charts will be displayed inline and exported.

---
*MAFS5370 Project 1 - Sicheng Zhu 21184536*
