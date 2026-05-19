from pathlib import Path
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from flightsim.axes3ds import Axes3Ds
from flightsim.world import World

from project3.PlanPath import PlanPath


def check(label, got, expected, tol=1e-6):
    ok = np.allclose(got, expected, atol=tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if not ok:
        print(f"       expected: {expected}")
        print(f"       got     : {got}")


def test_planning(map_name, methods, res_xy=0.2, res_z=0.2, margin=0.25):
    # ---------------------------------------------------------------------------
    # Load map
    # ---------------------------------------------------------------------------
    filename = f'{map_name}.json'
    file = Path(inspect.getsourcefile(lambda: 0)).parent.resolve() / '..' / 'maps' / filename
    my_world = World.from_file(file)

    start  = my_world.world['start']          # Start point, shape=(3,)
    goal   = my_world.world['goal']           # Goal point, shape=(3,)

    print(f"\nTesting map: {filename}")
    print("-" * 40)

    pp = PlanPath(my_world, res_xy=res_xy, res_z=res_z, margin=margin)

    # Colors and labels for each method
    method_styles = {
        'dijkstra': {'color': 'blue',  'label': 'Dijkstra'},
        'astar':    {'color': 'green', 'label': 'A*'},
    }

    # ---------------------------------------------------------------------------
    # Set up 3D figure for this map
    # ---------------------------------------------------------------------------
    fig = plt.figure(f'3D Path — {map_name}')
    ax  = Axes3Ds(fig)
    my_world.draw(ax)
    ax.set_zlim(0, 3)
    legend_handles = []

    for method in methods:
        print(f"\n  Method: {method}")
        style = method_styles.get(method, {'color': 'black', 'label': method})

        # Run planner
        if method == 'dijkstra':
            path = pp.runDijkstra()
        elif method == 'astar':
            path = pp.runAstar()
        else:
            print(f"  Unknown method: {method}")
            continue 
        # --- Check path exists ---
        check(f"{method}: path found", path is not None, True)
        if path is None:
            continue

        # --- Check start and goal are close to first and last waypoints ---
        thresholds = np.array([res_xy, res_xy, res_z])
        start_ok  = np.all(np.abs(path[0]  - pp.start) <= thresholds + 1e-9)
        goal_ok   = np.all(np.abs(path[-1] - pp.goal)  <= thresholds + 1e-9)
        print(f"[{'PASS' if start_ok else 'FAIL'}] {method}: path starts near start")
        print(f"[{'PASS' if goal_ok  else 'FAIL'}] {method}: path ends   near goal")

        # --- Check no waypoint lands in an occupied cell ---
        collision_free = True
        for wp in path:
            grid = pp.metric_to_grid(wp.reshape(1, 3))
            if pp.isOccupied[grid[0,0], grid[0,1], grid[0,2]]:
                collision_free = False
                print(f"[FAIL] {method}: collision at waypoint {wp}")
                break
        if collision_free:
            print(f"[PASS] {method}: path is collision free")

        # --- Draw path on 3D axes ---
        my_world.draw_line(ax, path, color=style['color'], linewidth=2)
        legend_handles.append(
            Line2D([], [], color=style['color'], linewidth=2, label=style['label'])
        )

    ax.plot([start[0]], [start[1]], [start[2]], 'o', markersize=12, markeredgewidth=3,
        markeredgecolor='lime', markerfacecolor='lime', alpha=0.8, zorder=5, label='Start')

    ax.plot([goal[0]], [goal[1]], [goal[2]], 'o', markersize=12, markeredgewidth=3,
            markeredgecolor='red', markerfacecolor='red', alpha=0.8, zorder=5, label='Goal')
    
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_zlim(0, 5)
    plt.tight_layout()

if __name__ == '__main__':
    methods = ['dijkstra', 'astar']
    for map_name in ['map0', 'map1', 'map2']:
        test_planning(map_name, methods)
    plt.show()

    methods = ['dijkstra', 'astar']
    map_name = 'map0'
    test_planning(map_name, methods, res_xy=0.2, res_z=0.2, margin=0.25)
    plt.show()