## This simulation uses a basic Q-learning approach where an agent learns to switch
## between three homeostatic baselines (-4, 0, 4) in response to non-normal sandpile
## avalanches.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## 1. Environment Parameters
baselines = np.array([-4.0, 0.0, 4.0])
n_states = 41  # Discretized space from -10 to 10
n_actions = 3  # Actions: [-1.0 (down), 0.0 (stay), 1.0 (up)]
alpha_rl = 0.1 # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Exploration rate

## Initialize Q-table: quality of actions in states
q_table = np.zeros((n_states, n_actions))

def get_state_idx(val):
    # Map continuous system state to discrete index
    return int(np.clip((val + 10) * 2, 0, n_states - 1))

## 2. Simulation State
system_state = 0.0
history_z = []
history_base = []
history_reward = []
time_steps = 300

## 3. Figure Setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line_z, = ax1.plot([], [], 'b-', lw=2, label='RL-AM Output Trajectory')
line_b, = ax1.plot([], [], 'g--', alpha=0.6, label='Active Baseline')
for b in baselines:
    ax1.axhline(y=b, color='gray', linestyle=':', alpha=0.3)
ax1.set_ylim(-10, 10)
ax1.set_xlim(0, time_steps)
ax1.set_title("Reinforcement Learning: Allostatic State Switching")
ax1.legend(loc='upper right')

line_r, = ax2.plot([], [], 'orange', label='Step Reward')
ax2.set_xlim(0, time_steps)
ax2.set_ylim(-5, 2)
ax2.set_title("Agent Reward Signal (Proximity to Nearest G)")
ax2.legend(loc='upper right')

def update(frame):
    global system_state, q_table

    ## A. Disturbance (Sandpile Avalanche - Power Law)
    if np.random.rand() < 0.05:
        magnitude = np.random.power(1.2) * 6.0
        disturbance = magnitude if np.random.rand() > 0.5 else -magnitude
    else:
        disturbance = 0.0

    ## B. Agent Action Selection (Epsilon-Greedy)
    state_idx = get_state_idx(system_state)
    if np.random.rand() < epsilon:
        action_idx = np.random.choice(n_actions)
    else:
        action_idx = np.argmax(q_table[state_idx])

    action_val = [ -1.5, 0.0, 1.5 ][action_idx]

    ## C. System Update (First-order dynamics)
    prev_state = system_state
    system_state = system_state + disturbance + action_val
    new_state_idx = get_state_idx(system_state)

    ## D. Allostatic Reward (Proximity to the NEAREST baseline)
    closest_base = baselines[np.argmin(np.abs(system_state - baselines))]
    distance = np.abs(system_state - closest_base)
    reward = 1.0 / (1.0 + distance**2) - 0.1 # Incentive for proximity

    ## E. Q-Table Update (Policy Refinement)
    best_future_q = np.max(q_table[new_state_idx])
    q_table[state_idx, action_idx] += alpha_rl * (reward + gamma * best_future_q - q_table[state_idx, action_idx])

    ## F. History Tracking
    history_z.append(system_state)
    history_base.append(closest_base)
    history_reward.append(reward)

    x_data = range(len(history_z))
    line_z.set_data(x_data, history_z)
    line_b.set_data(x_data, history_base)
    line_r.set_data(x_data, history_reward)

    return line_z, line_b, line_r

# Create the animation object
ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)

# Close the figure to prevent a static plot from being displayed alongside the animation
plt.close(fig)

# Display animation as HTML in the notebook
HTML(ani.to_jshtml())
