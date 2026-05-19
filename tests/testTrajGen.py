from pathlib import Path
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from flightsim.axes3ds import Axes3Ds
from flightsim.world import World

from project3.PlanPath import PlanPath
from project3.TrajGen import TrajGen
from project3.CubicPolyTraj import CubicPolyTraj

def check(label, got, expected, tol=1e-6):
    ok = np.allclose(got, expected, atol=tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if not ok:
        print(f"       expected: {expected}")
        print(f"       got     : {got}")


def test_trajGen():

    filename = 'map0.json'     
    file = Path(inspect.getsourcefile(lambda: 0)).parent.resolve() / '..' / 'maps' / filename
    my_world = World.from_file(file)

    start  = my_world.world['start']          # Start point, shape=(3,)
    goal   = my_world.world['goal']           # Goal point, shape=(3,)

    print(f"\nFinding Path map: {filename}")

    pp = PlanPath(my_world, res_xy=0.2, res_z=0.2, margin=0.25)
    path = pp.runAstar()

    check(f"A* path found", path is not None, True)
    if path is None:
        return
    tg = TrajGen(my_world, path, pp)
    pruned_waypoints = tg.prune_waypoints()
    print(f'Pruned waypoints: {len(pruned_waypoints)} / {len(path)} points')

    print(f"\nGenerating 3rd-order polynomials map: {filename}")
    
    # option 1 - Set velocities at everything waypoints as zero
    # pruned_waypoints_vel = np.zeros(pruned_waypoints.shape)

    # option 2 - Zero at endpoints, finite difference at interior points
    pruned_waypoints_vel = np.zeros(pruned_waypoints.shape)
    for i in range(1, len(pruned_waypoints) - 1):
        pruned_waypoints_vel[i] = (pruned_waypoints[i+1] - pruned_waypoints[i-1]) / 3.0 # increase the denominator to reduce the velocity


    coeffs, T = tg.generate_polynomials(pruned_waypoints, pruned_waypoints_vel)

    # ---------------------------------------------------------------------------
    # Sample trajectory and plot position, velocity, acceleration profiles
    # ---------------------------------------------------------------------------

    # Initialize Trajectory
    traj = CubicPolyTraj(coeffs, T)

    t_total = np.sum(T)
    t_samples = np.linspace(0, t_total, 500)

    pos  = np.zeros((len(t_samples), 3))
    vel  = np.zeros((len(t_samples), 3))
    acc  = np.zeros((len(t_samples), 3))

    for k, t in enumerate(t_samples):
        state      = traj.update(t)
        pos[k] = state['x']
        vel[k] = state['x_dot']
        acc[k] = state['x_ddot']

    # ---------------------------------------------------------------------------
    # Position vs. Time
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, num='Position vs Time')
    ax = axes[0]
    ax.plot(t_samples, pos[:, 0], 'r', linewidth=1)
    ax.legend(('x',), loc='upper right')
    ax.set_ylabel('Position [m]')
    ax.grid('major')
    ax.set_title('X')

    ax = axes[1]
    ax.plot(t_samples, pos[:, 1], 'g', linewidth=1)
    ax.legend(('y',), loc='upper right')
    ax.set_ylabel('Position [m]')
    ax.grid('major')
    ax.set_title('Y')

    ax = axes[2]
    ax.plot(t_samples, pos[:, 2], 'b', linewidth=1)
    ax.legend(('z',), loc='upper right')
    ax.set_ylabel('Position [m]')
    ax.grid('major')
    ax.set_title('Z')
    ax.set_xlabel('Time [s]')

    # ---------------------------------------------------------------------------
    # Velocity vs. Time
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, num='Velocity vs Time')
    ax = axes[0]
    ax.plot(t_samples, vel[:, 0], 'r', linewidth=1)
    ax.legend(('v_x',), loc='upper right')
    ax.set_ylabel('Velocity [m/s]')
    ax.grid('major')
    ax.set_title('V_X')

    ax = axes[1]
    ax.plot(t_samples, vel[:, 1], 'g', linewidth=1)
    ax.legend(('v_y',), loc='upper right')
    ax.set_ylabel('Velocity [m/s]')
    ax.grid('major')
    ax.set_title('V_Y')

    ax = axes[2]
    ax.plot(t_samples, vel[:, 2], 'b', linewidth=1)
    ax.legend(('v_z',), loc='upper right')
    ax.set_ylabel('Velocity [m/s]')
    ax.grid('major')
    ax.set_title('V_Z')
    ax.set_xlabel('Time [s]')

    plt.tight_layout()
    plt.show()
    

    # ---------------------------------------------------------------------------
    # Set up 3D figure
    # ---------------------------------------------------------------------------
    fig = plt.figure(f'3D Path — {filename}')
    ax  = Axes3Ds(fig)
    my_world.draw(ax)
    ax.set_zlim(0, 3)
    legend_handles = []
    ax.plot(path[:, 0], path[:, 1], path[:, 2],
            color='royalblue', linewidth=2, zorder=7, label='Path')
    # Draw pruned waypoints as scatter points
    ax.scatter(pruned_waypoints[:, 0], pruned_waypoints[:, 1], pruned_waypoints[:, 2],
               color='orange', s=50, zorder=6, label='Pruned Waypoints')
    # Draw polynomial trajectory
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color='magenta', linewidth=2, zorder=7, label='Polynomial Trajectory')
    
    ax.plot([start[0]], [start[1]], [start[2]], 'o', markersize=12, markeredgewidth=3,
            markeredgecolor='lime', markerfacecolor='lime', alpha=0.8, zorder=5, label='Start')
    ax.plot([goal[0]], [goal[1]], [goal[2]], 'o', markersize=12, markeredgewidth=3,
            markeredgecolor='red', markerfacecolor='red', alpha=0.8, zorder=5, label='Goal')
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_title(f'3D Path & Polynomial Trajectory — {filename}')
    ax.legend(loc='upper right')
    ax.set_zlim(0, 5)
    plt.tight_layout()

# ---------------------------------------------------------------------------
# Run for each map
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    test_trajGen()
    plt.show()