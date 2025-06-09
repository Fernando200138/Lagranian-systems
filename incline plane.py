import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Parameters
m = 1    # mass of the block
M = 1    # mass of the inclined plane
theta_deg = 45
theta = np.radians(theta_deg)  # convert angle to radians
g = 1    # use g = 1 for normalized gravity (can scale later)

# Equations of motion as a system of first-order ODEs
def equations(t, y):
    x1, x1_dot, x2, x2_dot = y

    # Compute accelerations
    x2_ddot = -np.sin(theta)*g - (m * x2_dot * np.cos(theta) / (M + m))  # approximate x1'' with x2' version
    x1_ddot = -(m * x2_ddot * np.cos(theta)) / (M + m)

    return [x1_dot, x1_ddot, x2_dot, x2_ddot]

# Initial conditions
x1_0 = 0
x1_dot_0 = 0
x2_0 = 5
x2_dot_0 = 0
y0 = [x1_0, x1_dot_0, x2_0, x2_dot_0]

# Time span
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 300)

# Solve ODE
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# Extract solutions
x1_vals = sol.y[0]
x2_vals = sol.y[2]
t_vals = sol.t

# Compute block's position in 2D space
x_block = x1_vals + x2_vals * np.cos(theta)
y_block = x2_vals * np.sin(theta)

# Animation
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-2, np.max(x_block) + 2)
ax.set_ylim(-1, np.max(y_block) + 1)
ax.set_aspect('equal')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Block on Moving Inclined Plane")

# Elements
incline_line, = ax.plot([], [], 'k-', lw=3)
block_dot, = ax.plot([], [], 'ro', markersize=10)
plane_rect = plt.Rectangle((0, -0.2), 3, 0.2, fc='gray', ec='black')
ax.add_patch(plane_rect)


def init():
    incline_line.set_data([], [])
    block_dot.set_data([], [])
    plane_rect.set_xy((x1_vals[0], -0.2))
    return incline_line, block_dot, plane_rect


def update(frame):
    x1 = x1_vals[frame]
    x_b = x_block[frame]
    y_b = y_block[frame]

    # Incline endpoints (approximate length)
    L = 6
    x_start = x1
    x_end = x1 + L * np.cos(theta)
    y_start = 0
    y_end = L * np.sin(theta)

    incline_line.set_data([x_start, x_end], [y_start, y_end])
    block_dot.set_data(x_b, y_b)
    plane_rect.set_xy((x1, -0.2))

    return incline_line, block_dot, plane_rect


ani = FuncAnimation(fig, update, frames=len(t_vals), init_func=init, blit=True, interval=30)
plt.show()
