import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import ipywidgets as widgets
from IPython.display import display

# -----------------------------
# 1. System Parameters
# -----------------------------
baselines = [-4.0, 0.0, 4.0]
current_baseline = 0.0
system_state = 0.0

time_steps = 1000

# Memory slider (ipywidgets)
memory_slider = widgets.IntSlider(
    value=20,
    min=1,
    max=100,
    step=1,
    description='Memory Window',
    continuous_update=True
)

display(memory_slider)

# History storage
history_z = []
history_base = []
history_d = []

# -----------------------------
# 2. Plot Setup
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

line_z, = ax1.plot([], [], 'b-', lw=1.5, label='AM Trajectory (Z)')
line_b, = ax1.plot([], [], 'g--', alpha=0.7, label='Active Baseline')
point_z, = ax1.plot([], [], 'bo')

for b in baselines:
    ax1.axhline(y=b, color='gray', linestyle=':', alpha=0.4)

ax1.set_xlim(0, 200)
ax1.set_ylim(-10, 10)
ax1.set_title("Allostasis Machine: Stochastic Switching with Memory Window")
ax1.legend(loc='upper right')

line_d, = ax2.plot([], [], 'r-', alpha=0.5, label='Sandpile Avalanches (D)')
ax2.set_xlim(0, 200)
ax2.set_ylim(-6, 6)
ax2.set_title("Perturbation Magnitude (Stochastic Sequence)")
ax2.legend(loc='upper right')


# -----------------------------
# 3. Update Function
# -----------------------------
def update(frame):
    global system_state, current_baseline

    # A. Read memory window from ipywidget
    mem_window = int(memory_slider.value)

    # B. Stochastic perturbation (sandpile-like)
    if np.random.rand() < 0.06:
        magnitude = np.random.power(1.2) * 6.0
        disturbance = magnitude if np.random.rand() > 0.5 else -magnitude
    else:
        disturbance = 0.0
    history_d.append(disturbance)

    # C. Allostatic switching
    closest_baseline = min(baselines, key=lambda b: abs(system_state - b))
    if closest_baseline != current_baseline:
        current_baseline = closest_baseline

    # D. Memory-based correction
    recent_p = history_d[-mem_window:]
    memory_correction = np.mean(recent_p) if recent_p else 0.0

    error = system_state - current_baseline
    recovery = -0.25 * error - 0.4 * memory_correction

    # E. Update trajectory
    system_state = system_state + disturbance + recovery
    history_z.append(system_state)
    history_base.append(current_baseline)

    # Sliding window
    start_idx = max(0, len(history_z) - 300)
    x_data = range(len(history_z[start_idx:]))

    line_z.set_data(x_data, history_z[start_idx:])
    line_b.set_data(x_data, history_base[start_idx:])
    line_d.set_data(x_data, history_d[start_idx:])
    point_z.set_data([len(x_data)-1], [system_state])

    ax1.set_xlim(0, 300)
    ax2.set_xlim(0, 300)

    return line_z, line_b, line_d, point_z


# -----------------------------
# 4. Animation
# -----------------------------
ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)

plt.close(fig)
HTML(ani.to_jshtml())
