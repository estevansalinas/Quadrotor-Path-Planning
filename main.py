from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import inspect

from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World
from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params

#################### From Part 2 Select a Controller ####################
from project2.controllers.pd_controller import PDControl
from project2.controllers.se3_controller import SE3Control

from project2.trajectories.HelixTraj import HelixTraj

##################### Path Planning #####################################
from project3.PlanPath import PlanPath
from project3.TrajGen import TrajGen
from project3.CubicPolyTraj import CubicPolyTraj

def main():
    # Load the test example.
    filename = 'map2.json'
    file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / 'maps' / filename
    my_world = World.from_file(file)          # World boundary and obstacles.
    start  = my_world.world['start']          # Start point, shape=(3,)
    goal   = my_world.world['goal']           # Goal point, shape=(3,)

    # This object defines the quadrotor dynamical model and should not be changed.
    quadrotor = Quadrotor(quad_params)
    robot_radius = 0.25

    # Select Your Controller.
    my_controller = PDControl(quad_params)

    # Run Path Planning
    # method = 'dijkstra'
    method = 'astar'
    print(f"\n Finding Path for Map: {filename} in {method} Search Algorithm!")

    pp = PlanPath(my_world, res_xy=0.2, res_z=0.2, margin=robot_radius)

    # Run planner
    if method == 'dijkstra':
        path = pp.runDijkstra()
    elif method == 'astar':
        path = pp.runAstar()
    else:
        print(f"  Unknown method: {method}")
        return

    # --- Check path exists ---
    if path is None:
        print("Path not found.")
        return
    else: 
        print("Path found.")

    print(f"\n Generate Multi-Segment Cubic Polynomial Trajectory for Map: {filename} in {method} Search Algorithm!")
    tg = TrajGen(my_world, path, pp)
    pruned_waypoints = tg.prune_waypoints()
    print(f'Pruned waypoints: {len(pruned_waypoints)} / {len(path)} points')

    # option 1 - Set velocities at everything waypoints as zero
    # pruned_waypoints_vel = np.zeros(pruned_waypoints.shape)

    # option 2 - Zero at endpoints, finite difference at interior points
    pruned_waypoints_vel = np.zeros(pruned_waypoints.shape)
    for i in range(1, len(pruned_waypoints) - 1):
        pruned_waypoints_vel[i] = (pruned_waypoints[i+1] - pruned_waypoints[i-1]) / 10.0  # increase the denominator to reduce the velocity


    coeffs, T = tg.generate_polynomials(pruned_waypoints, pruned_waypoints_vel)
    # Initialize Trajectory
    my_trajectory = CubicPolyTraj(coeffs, T)

    # Get starting flat output from trajectory
    flat0 = my_trajectory.update(0)

    # Set simulation parameters
    t_final = np.sum(T) + 2.0    # total trajectory duration, s

    initial_state = {
        'x': tuple(flat0['x']),      # start position from trajectory
        'v': tuple(flat0['x_dot']),  # start velocity from trajectory
        'q': (0, 0, 0, 1),           # neutral orientation
        'w': (0, 0, 0)               # zero angular velocity
    }
    print("initial_state = ", initial_state)

    # Perform simulation.
    #
    # This function performs the numerical simulation.  It returns arrays reporting
    # the quadrotor state, the control outputs calculated by your controller, and
    # the flat outputs calculated by you trajectory.
    print()
    print('Simulate.')
    (sim_time, state, control, flat, exit_status) = simulate(initial_state,
                                                  quadrotor,
                                                  my_controller,
                                                  my_trajectory,
                                                  t_final, terminate=False, vio = None)
    print(exit_status.value)

    # Position vs. Time
    (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Position vs Time')
    x = state['x']
    x_T = flat['x']
    ax = axes[0]
    ax.plot(sim_time, x[:,0], 'r', sim_time, x_T[:,0], 'k--', linewidth=1, alpha=0.6)
    ax.legend(('x', 'x_T'), loc='upper right')
    ax.set_ylabel('Position [m]')
    ax.grid('major')
    ax.set_title('X ')


    ax = axes[1]
    ax.plot(sim_time, x[:,1], 'g', sim_time, x_T[:,1], 'k--', linewidth=1, alpha=0.6)
    ax.legend(('y', 'y_T'), loc='upper right')
    ax.set_ylabel('Position [m]')
    ax.grid('major')
    ax.set_title('Y ')

    ax = axes[2]
    ax.plot(sim_time, x[:,2], 'b', sim_time, x_T[:,2], 'k--', linewidth=1, alpha=0.6)
    ax.legend(('z', 'z_T'), loc='upper right')
    ax.set_ylabel('Position [m]')
    ax.grid('major')
    ax.set_title('Z ')
    ax.set_xlabel('Time [s]')

    # Velocity vs. Time
    (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Velocity vs Time')
    v = state['v']
    v_T = flat['x_dot']
    ax = axes[0]
    ax.plot(sim_time, v[:,0], 'r', sim_time, v_T[:,0], 'k--', linewidth=1, alpha=0.6)
    ax.legend(('v_x', 'v_{x,T}'), loc='upper right')
    ax.set_ylabel('Velocity [m/s]')
    ax.grid('major')
    ax.set_title('V_X ')


    ax = axes[1]
    ax.plot(sim_time, v[:,1], 'g', sim_time, v_T[:,1], 'k--', linewidth=1, alpha=0.6)
    ax.legend(('v_y', 'v_{y,T}'), loc='upper right')
    ax.set_ylabel('Velocity [m/s]')
    ax.grid('major')
    ax.set_title('V_Y ')

    ax = axes[2]
    ax.plot(sim_time, v[:,2], 'b', sim_time, v_T[:,2], 'k--', linewidth=1, alpha=0.6)
    ax.legend(('v_z', 'v_{z,T}'), loc='upper right')
    ax.set_ylabel('Velocity [m/s]')
    ax.grid('major')
    ax.set_title('V_Z ')
    ax.set_xlabel('Time [s]')


    # Commands vs. Time
    (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
    s = control['cmd_motor_speeds']
    ax = axes[0]
    ax.plot(sim_time, s[:,0], 'r', sim_time, s[:,1], 'g', sim_time, s[:,2], 'b', sim_time, s[:,3], 'k', linewidth=1, alpha=0.6)
    ax.legend(('1', '2', '3', '4'), loc='upper right')
    ax.set_ylabel('Motor Speeds [rad/s]')
    ax.grid('major')
    ax.set_title('Commands')
    M = control['cmd_moment']
    ax = axes[1]
    ax.plot(sim_time, M[:,0], 'r', sim_time, M[:,1], 'g', sim_time, M[:,2], 'b', linewidth=1, alpha=0.6)
    ax.legend(('x', 'y', 'z'), loc='upper right')
    ax.set_ylabel('Moment [N*m]')
    ax.grid('major')
    T = control['cmd_thrust']
    ax = axes[2]
    ax.plot(sim_time, T, 'k', linewidth=1, alpha=0.6)
    ax.set_ylabel('Thrust [N]')
    ax.set_xlabel('Time [s]')
    ax.grid('major')


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
    # Draw desired and actual polynomial trajectory
    ax.plot(x[:, 0], x[:, 1], x[:, 2],
            color='red', linewidth=2, zorder=7, label='p')
    ax.plot(x_T[:, 0], x_T[:, 1], x_T[:, 2],
            color='black', linewidth=2, zorder=7, label='p_T')
    ax.plot([start[0]], [start[1]], [start[2]], 'o', markersize=12, markeredgewidth=3,
            markeredgecolor='lime', markerfacecolor='lime', alpha=0.8, zorder=5, label='Start')
    ax.plot([goal[0]], [goal[1]], [goal[2]], 'o', markersize=12, markeredgewidth=3,
            markeredgecolor='red', markerfacecolor='red', alpha=0.8, zorder=5, label='Goal')
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_title(f'3D Path & Polynomial Trajectory — {filename}')
    ax.legend(loc='upper right')
    ax.set_zlim(0, 5)
    plt.tight_layout()

    plt.show()


    # animation
    R = Rotation.from_quat(state['q']).as_matrix()
    ani = animate(sim_time, state['x'], R, zlim_pos=3, world=my_world, filename=None)
    plt.show()

if __name__ == '__main__':
    main()
