## Project 1 for MAFS5370

## Sicheng Zhu 21184536

Using seven algorithms to complete this problem:
Greedy, Q-Learning(using adam to tune the learning rate alpha), Reinforce, Actor-Critic, A3C, PPO and TRPO

Discretize the continuous asset adjustments: each asset's weight change is from -0.1 to 0.1 in steps of 0.01, with the constraint that the total sum of absolute adjustments across all assets is less than 0.1.

Initial parameters:

Time horizon T = 8

Number of assets = 3+1(risk free asset)

Expected returns = [-0.5, 0.5, 0.0]

Return variances = [0.25, 0.25, 0.5]

Risk-free rate = 0.02

Initial weights = [0.25, 0.25, 0.25, 0.25]

Risk aversion parameter = 1.0

Allow short selling = False

Using 3000 episodes to train the model and using 500 episodes to test the result, final wealth is calculated based on the average utility of the test period.
