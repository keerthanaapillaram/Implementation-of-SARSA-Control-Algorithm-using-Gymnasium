# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the SARSA control algorithm in the Gymnasium FrozenLake-v1 environment. The agent must learn an optimal action-selection policy through interaction with the environment. For this experiment, a customized 4×4 FrozenLake map is used with a modified starting position and goal position. The agent learns to navigate from the starting state to the goal while avoiding hole states

## Software Requirements
```
Python 3.x
Jupyter Notebook / Google Colab
NumPy
Matplotlib
Gymnasium
```


## Environment Description
A customized 4×4 FrozenLake environment is used for the experiment.
```
F F H F
F S F F
H F F F
F F H G
```
Where:

S represents the starting state.
G represents the goal state.
F represents a safe frozen tile.
H represents a hole.

The starting state is state 5, and the goal state is state 15.

The environment is created using:

custom_map = [
    "FFHF",
    "FSFF",
    "HFFF",
    "FFHG"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=False
)

The is_slippery=False setting makes the environment deterministic, so the agent's selected action produces the corresponding movement without random slipping.


## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

1. Create the customized FrozenLake environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate α, discount factor γ, and exploration parameters.
4. Reset the environment at the beginning of every episode.
5. Select the initial action using the epsilon-greedy policy.
6. Execute the selected action.
7. Observe the next state and reward.
8. If the episode has not terminated, select the next action using the same epsilon-greedy policy.
9. Update the Q-value using the SARSA update equation.
10. Move to the next state and action.
11. Repeat until the episode terminates or the maximum number of steps is reached.
12. Decrease epsilon gradually to reduce exploration.
13. Repeat the process for 10,000 episodes.
14. Extract the learned policy from the final Q-table.
15. Calculate the state-value function and average reward.
16. Plot the learning curve.

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Customized FrozenLake Environment
# -------------------------------------------------

custom_map = [
    "FFHF",
    "FSFF",
    "HFFF",
    "FFHG"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=False
)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros(
    (env.observation_space.n,
     env.action_space.n)
)

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])


# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    action = epsilon_greedy_action(
        state,
        epsilon
    )

    total_reward = 0

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # If episode ends
        if terminated or truncated:

            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

            break

        # Select next action using epsilon-greedy policy
        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # SARSA update
        Q[state, action] += alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        state = next_state
        action = next_action

    episode_rewards.append(total_reward)


# -------------------------------------------------
# State-Value Function
# -------------------------------------------------

state_values = np.max(Q, axis=1)

# -------------------------------------------------
# Learned Policy
# -------------------------------------------------

learned_policy = np.argmax(Q, axis=1)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - Customized FrozenLake")

plt.grid(True)
plt.show()

env.close()




```
---

## Output

Final Q-table:

<img width="656" height="326" alt="image" src="https://github.com/user-attachments/assets/aa3e9872-f6cf-4f16-ad31-0445b127d18d" />


Estimated State-Value Function:

<img width="551" height="112" alt="image" src="https://github.com/user-attachments/assets/22bec211-5f6d-4937-85e4-252a64a6cae3" />


Learned Policy:

<img width="522" height="110" alt="image" src="https://github.com/user-attachments/assets/c58a27e3-0e94-4772-928a-e7e979c410ac" />


Average reward over last 1000 episodes: 

<img width="561" height="40" alt="image" src="https://github.com/user-attachments/assets/33356e0a-d0e9-4b8e-bd12-93189c5c4204" />


<img width="918" height="477" alt="image" src="https://github.com/user-attachments/assets/9d5a9d41-d880-4cce-8d7d-722661249396" />


## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. A customized 4×4 environment was used by changing the starting position to state 5 and defining state 15 as the goal. Through repeated interaction with the environment, the agent learned an action-value function and an epsilon-greedy policy for navigating through the environment while avoiding hole states.


## Inference

The experiment shows that SARSA can learn a control policy through trial-and-error interaction with the environment. Initially, the agent performs more exploration because epsilon is high. As training progresses, epsilon decreases and the agent increasingly uses the learned Q-values to select actions. The learned policy therefore improves its ability to navigate toward the goal while avoiding the holes in the customized FrozenLake environment.


