## How do disturbances (D) affect the stability of system outcomes (Z)?

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## Simulation Parameters
target_g = 0.0          # Goal setpoint (G)
system_state = 0.0      # Current state (S)
k_p = 0.4               # Proportional Gain (Regulatory correction)
alpha = 1.0             # Power law exponent for disturbance [7]
burst_probability = 0.05 # Probability of an avalanche occurring at any step
time_steps = 300

history_s = []
history_r = []
history_d = []

## Setup the figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line_s, = ax1.plot([], [], 'b-', label='System State (S)')
point_s, = ax1.plot([], [], 'bo')
ax1.axhline(y=target_g, color='g', linestyle='--', label='Target G')
ax1.set_xlim(0, time_steps)
ax1.set_ylim(-6, 6)
ax1.set_title("Regulation Against Sandpile Avalanches (Non-Normal Disturbances)")
ax1.legend(loc='upper right')

## Plot for disturbances and regulatory response
line_dist, = ax2.plot([], [], 'k-', alpha=0.3, label='Perturbations (D)')
line_reg, = ax2.plot([], [], 'r-', label='Regulator Output (R)')
ax2.set_xlim(0, time_steps)
ax2.set_ylim(-6, 6)
ax2.set_title("Perturbation Magnitude vs. Regulatory Correction")
ax2.legend(loc='upper right')

def update(frame):
    global system_state

    ## 1. Non-normally distributed perturbations (D)
    ## Using a random trigger for non-uniform intervals
    if np.random.rand() < burst_probability:
        # Magnitude based on power law distribution [7]
        magnitude = np.random.power(alpha) * 5.0
        # Randomize direction
        disturbance = magnitude if np.random.rand() > 0.5 else -magnitude
    else:
        disturbance = 0.0

    ## 2. Regulator Action (R)
    ## The regulator uses its internal model to calculate a correction [8, 9]
    error = system_state - target_g
    regulatory_action = -k_p * error

    ## 3. System Update (S)
    ## Joint determination of outcome Z via D and R [10, 11]
    system_state = system_state + disturbance + regulatory_action

    history_s.append(system_state)
    history_r.append(regulatory_action)
    history_d.append(disturbance)

    x_data = range(len(history_s))
    line_s.set_data(x_data, history_s)
    point_s.set_data([frame], [system_state])
    line_dist.set_data(x_data, history_d)
    line_reg.set_data(x_data, history_r)

    return line_s, point_s, line_dist, line_reg

## Create the animation object
ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)

## Close the figure to prevent a static plot from being displayed alongside the animation
plt.close(fig)

## Display animation as HTML in the notebook
HTML(ani.to_jshtml())
