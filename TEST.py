import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

g=9.81
m=1
k=0.1
def system(t,y):
    x1,v1,x2,v2=y
    dx1dt = v1
    dv1dt= (g*x2*np.sin(x1)-v2*x2*v1)/x2**2
    dx2dt = v2
    dv2dt = x2*v1**2-g*np.cos(x1)-k*x2
    return [dx1dt,dv1dt,dx2dt,dv2dt]

y0 = [np.pi/4,0.3,3,0.0]

t_span = (0,100)
t_eval = np.linspace(0,100,100)
sol = solve_ivp(system,t_span,y0, t_eval = t_eval,method='RK45')

X = []
Y = []
for i in range(len(sol.t)):
    X.append(sol.y[2][i]*np.sin(sol.y[0][i]))
    Y.append(sol.y[2][i]*np.cos(sol.y[0][i]))

fig, ax = plt.subplots()
ax.set_xlim(min(X) - 1, max(X) + 1)
ax.set_ylim(min(Y) - 1, max(Y) + 1)
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
    point.set_data(X[frame], Y[frame])  # Update point position
    trail.set_data(X[:frame+1], Y[:frame+1])  # Update trajectory trail
    spring.set_data([x_anchor, X[frame]], [y_anchor, Y[frame]])  # Update spring line

    return point, trail,spring
t=sol.t
# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)

# Show animation
plt.show()


#plt.plot(sol.t, sol.y[0], label="x1(t)")
#plt.plot(sol.t, sol.y[2], label="x2(t)")
#plt.xlabel("Time")
#plt.ylabel("x1, x2")
#plt.legend()
#plt.show()