import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Ladder properties
L = 1.0  # Length of the ladder
m=1
g=9.8
# Time array
t = np.linspace(0.3, 0.5, 100)  # Adjust 10 and 500 as needed

# Your solution: theta(t)
# Example: simple assumption (not real physics) -- you should replace this!
theta = [np.pi,0]  # starts vertical, slides down
print('Initial set up succesful....')
def system(t,y):
    x1,v1 = y
    dx1dt = v1
    dv1dt = (4/L)*g*np.cos(x1)
    return [dx1dt, dv1dt]
sol = solve_ivp(system, [0, t[-1]], theta, t_eval=t)
# Compute endpoints
print('Solving......')
x_bottom = L * np.cos(sol.y[0])
y_top = L * np.sin(sol.y[0])

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-0.1, L + 0.1)
ax.set_ylim(-0.1, L + 0.1)
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    # Coordinates of the ladder at this frame
    xdata = [x_bottom[frame], 0]
    ydata = [0, y_top[frame]]
    line.set_data(xdata, ydata)
    return line,

ani = FuncAnimation(fig, update, frames=len(t),
                    init_func=init, blit=True, interval=20)

plt.show()
