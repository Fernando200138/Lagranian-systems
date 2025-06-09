import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

g = 9.81
m = 1
a=1
b=4
w = np.pi/20
def system(t,y):
    x1,v1 = y

    dx1dt = v1
    dv1dt = (a*w**2/b**2)*np.cos(x1-w*t)-g/b*np.sin(x1)
    return [dx1dt,dv1dt]

y0 = [np.pi,0]
t_span = (0,1000)
t_eval = np.linspace(0,1000,10000)

sol= solve_ivp(system,t_span,y0,t_eval=t_eval,method='RK45')

X = []
Y = []

for i in range(len(sol.t)):

    X.append(a*np.cos(w*sol.t[i])+b*np.sin(sol.y[0][i]))
    Y.append(a*np.sin(w*sol.t[i])-b*np.cos(sol.y[0][i]))

fig, ax = plt.subplots()
ax.set_xlim(min(X) - 1, max(X) + 3)
ax.set_ylim(min(Y) - 1, max(Y) + 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Trajectory Animation")

# Initialize the point
point, = ax.plot([], [], 'ro', markersize=8)  # Red point
trail, = ax.plot([], [], 'b-', lw=1)  # Blue trail
spring, = ax.plot([], [], 'k-', lw=2)  # Black line for the spring
x_anchor, y_anchor = 0, 0  

# Update function for animation
def update(frame):
    # Compute anchor position at this frame
    x_anchor = a * np.cos(w * t[frame])
    y_anchor = a * np.sin(w * t[frame])

    # Update the bob (mass) position
    point.set_data(X[frame], Y[frame])
    # Update the trail
    trail.set_data(X[:frame+1], Y[:frame+1])
    # Update the spring (pendulum rod) from anchor to bob
    spring.set_data([x_anchor, X[frame]], [y_anchor, Y[frame]])

    return point, trail, spring
circle = plt.Circle((0, 0), a, color='gray', fill=False, linestyle='--')
ax.add_artist(circle)

t=sol.t
# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)

# Show animation
plt.show()