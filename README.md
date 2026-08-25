# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the SARSA control algorithm in the Gymnasium FrozenLake-v1 environment. The agent must learn an effective action-selection policy through interaction with the environment. A customized 4×4 FrozenLake map is used with the starting position at state 5 and the goal position at state 15. The agent learns to navigate from the starting state to the goal while avoiding hole states.

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

The environment consists of safe frozen tiles (`F`), hole states (`H`), a starting state (`S`), and a goal state (`G`).

```text
F F H F
F S F F
H F F F
F F H G
```

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

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned an action-value function and an epsilon-greedy policy for navigating from state 5 to the goal state 15 while avoiding the hole states. The learned Q-table and policy demonstrate the agent's ability to select suitable actions based on its experience.


## Inference

The experiment shows that SARSA can learn an effective control policy through trial-and-error interaction with the environment. With a fixed epsilon, the exploration rate remains constant throughout training, whereas with epsilon decay, exploration is high initially and gradually decreases as training progresses. In this experiment, epsilon decay allows the agent to explore different actions in the beginning and increasingly exploit the learned Q-values later. This helps the agent learn a suitable path toward the goal while avoiding the holes in the customized FrozenLake environment.

The experiment shows that SARSA can learn a control policy through trial-and-error interaction with the environment. Initially, the agent performs more exploration because epsilon is high. As training progresses, epsilon decreases and the agent increasingly uses the learned Q-values to select actions. The learned policy therefore improves its ability to navigate toward the goal while avoiding the holes in the customized FrozenLake environment.


