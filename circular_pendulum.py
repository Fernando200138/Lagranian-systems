import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

g = 9.81
m = 1
b = 3

def system(t,y):
    x1,p1,x2,p2 = y

    dx1dt= (1/(m*b**2))*p1
    dx2dt = (1/(np.sin(x1)*m*b**2))
    dp1dt= (np.cos(x1)*p2**2)/(m*b**2*(np.sin(x1))**3)+m*g*b*np.sin(x1)
    dp2dt=0
    return [dx1dt,dp1dt,dx2dt,dp2dt]

y0 = [np.pi,np.pi,0,0]
t_span = (0,1000)
t_eval = np.linspace(0,1000,10000)
sol= solve_ivp(system,t_span,y0,t_eval=t_eval,method='RK45')

X = []
Y = []
Z = []
for i in range(len(sol.t)):
    X.append(b*np.cos(sol.y[2][i])*np.sin(sol.y[0][i]))
    Y.append(b*np.sin(sol.y[2][i])*np.sin(sol.y[0][i]))
    Z.append(b*np.cos(sol.y[0][i]))


data = np.column_stack((X, Y, Z))


# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the pendulum parts
pivot = np.array([0, 0, 0])  # Fixed pivot point
mass, = ax.plot([], [], [], 'ro', markersize=8)  # Moving mass
rod, = ax.plot([], [], [], 'k-', linewidth=2)  # Rod connecting mass to pivot
trail, = ax.plot([], [], [], 'b-', alpha=0.6)  # Trail line

# Set axis limits
ax.set_xlim([min(X), max(X)])
ax.set_ylim([min(Y), max(Y)])
ax.set_zlim([min(Z), max(Z)])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Circular Pendulum")
trail_length = 30  # Number of past points to keep
trail_data = []  # List to store past positions
# Update function for animation
def update(frame):
    x, y, z = data[frame]
    mass.set_data([x], [y])
    mass.set_3d_properties([z])
    rod.set_data([pivot[0], x], [pivot[1], y])
    rod.set_3d_properties([pivot[2], z])
    trail_data.append([x, y, z])  # Add current position
    if len(trail_data) > trail_length:
        trail_data.pop(0)  # Keep only last 'trail_length' points

    # Convert trail data to arrays for plotting
    trail_array = np.array(trail_data)
    trail.set_data(trail_array[:, 0], trail_array[:, 1])
    trail.set_3d_properties(trail_array[:, 2])
    return mass, rod, trail

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(data), interval=50, blit=True)

plt.show()