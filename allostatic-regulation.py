## Create an interactive simulation that demonstrates regulation against sandpile
## avalanches for allostasis, or several homeostatic states. Show that the AM
## Output Trajectory can switch between different homeostatic states based on the
## magnitude of the perturbation. If the trajectory gets closer to a neighboring
## homeostatic baseline or rather than the original homeostatic baseline, then
## switch to that as the new baseline.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## Simulation Parameters
baselines = [-4.0, 0.0, 4.0]  # Multiple homeostatic states (G)
current_baseline = 0.0        # Initial regulated state
system_state = 0.0            # Initial output trajectory (Z)
recovery_gain = 0.25          # Recovery capacity (tr)
alpha = 1.1                   # Power law shape parameter for avalanches
burst_prob = 0.05             # Frequency of perturbations
time_steps = 300

# Data storage
history_z = []      # Output Trajectory
history_base = []   # Current active baseline
history_d = []      # Disturbances (D)

## Setup figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line_z, = ax1.plot([], [], 'b-', lw=2, label='AM Output Trajectory (Z)')
line_b, = ax1.plot([], [], 'g--', alpha=0.7, label='Active Baseline')
point_z, = ax1.plot([], [], 'bo')
for b in baselines:
    ax1.axhline(y=b, color='gray', linestyle=':', alpha=0.4)

ax1.set_xlim(0, time_steps)
ax1.set_ylim(-8, 8)
ax1.set_title("Allostatic Regulation: State Switching via Sandpile Avalanches")
ax1.legend(loc='upper right')

line_d, = ax2.plot([], [], 'r-', alpha=0.6, label='Avalanche Magnitude (D)')
ax2.set_xlim(0, time_steps)
ax2.set_ylim(-6, 6)
ax2.set_title("Stochastic Perturbation Sequence")
ax2.legend(loc='upper right')

def update(frame):
    global system_state, current_baseline

    ## 1. Generate Non-Normal Disturbance (Sandpile Avalanche)
    if np.random.rand() < burst_prob:
        # Scale-free power law noise
        magnitude = np.random.power(alpha) * 5.0
        disturbance = magnitude if np.random.rand() > 0.5 else -magnitude
    else:
        disturbance = 0.0

    ## 2. Allostatic Switching Logic
    ## Identify the closest neighboring baseline to the current state
    closest_baseline = min(baselines, key=lambda b: abs(system_state - b))

    ## If closer to a different baseline, switch (Allostatic Drift)
    if closest_baseline != current_baseline:
        current_baseline = closest_baseline

    ## 3. Regulatory Action (Recovery towards active baseline)
    error = system_state - current_baseline
    recovery = -recovery_gain * error

    ## 4. System Update
    ## Joint determination of Z by R and D
    system_state = system_state + disturbance + recovery

    history_z.append(system_state)
    history_base.append(current_baseline)
    history_d.append(disturbance)

    ## Update visuals
    x_data = range(len(history_z))
    line_z.set_data(x_data, history_z)
    line_b.set_data(x_data, history_base)
    point_z.set_data([frame], [system_state])
    line_d.set_data(x_data, history_d)

    return line_z, line_b, point_z, line_d

## Create the animation object
ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)

## Close the figure to prevent a static plot from being displayed alongside the animation
plt.close(fig)

## Display animation as HTML in the notebook
HTML(ani.to_jshtml())
