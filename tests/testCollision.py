from pathlib import Path
import inspect
import numpy as np
from flightsim.world import World
from project3.PlanPath import PlanPath

def check(label, got, expected, tol=1e-6):
    ok = np.allclose(got, expected, atol=tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if not ok:
        print(f"       expected: {expected}")
        print(f"       got     : {got}")

def test_isOccupied(map_name):
    # ---------------------------------------------------------------------------
    # Load map
    # ---------------------------------------------------------------------------
    filename = f'{map_name}.json'
    file = Path(inspect.getsourcefile(lambda: 0)).parent.resolve() / '..' / 'maps' / filename
    my_world = World.from_file(file)
    print(f"\nTesting map: {filename}")
    print("-" * 40)
    pp = PlanPath(my_world, res_xy=0.2, res_z=0.2, margin=0.25)
    margin = pp.margin

    # ---------------------------------------------------------------------------
    # Per-block tests
    # ---------------------------------------------------------------------------
    for i, block in enumerate(pp.blocks):
        xmin, xmax, ymin, ymax, zmin, zmax = block
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        # OCCUPIED: centre of block
        grid = pp.metric_to_grid(np.array([[xmid, ymid, zmid]]))
        check(f"isOccupied: block{i} centre ({xmid:.2f},{ymid:.2f},{zmid:.2f})",
              pp.isOccupied[grid[0,0], grid[0,1], grid[0,2]], True)

        # OCCUPIED: inside margin (shift y by margin/2 outside ymin face)
        y_margin = ymin - margin / 2
        grid = pp.metric_to_grid(np.array([[xmid, y_margin, zmid]]))
        check(f"isOccupied: block{i} margin  ({xmid:.2f},{y_margin:.2f},{zmid:.2f})",
              pp.isOccupied[grid[0,0], grid[0,1], grid[0,2]], True)

        # FREE: just outside margin
        y_free = ymin - margin - pp.res_xyz[1]
        grid = pp.metric_to_grid(np.array([[xmid, y_free, zmid]]))
        check(f"isOccupied: block{i} free    ({xmid:.2f},{y_free:.2f},{zmid:.2f})",
              pp.isOccupied[grid[0,0], grid[0,1], grid[0,2]], False)

    # ---------------------------------------------------------------------------
    # Start and goal must always be free
    # ---------------------------------------------------------------------------
    grid = pp.metric_to_grid(np.array([pp.start]))
    check(f"isOccupied: start free {pp.start}",
          pp.isOccupied[grid[0,0], grid[0,1], grid[0,2]], False)

    grid = pp.metric_to_grid(np.array([pp.goal]))
    check(f"isOccupied: goal  free {pp.goal}",
          pp.isOccupied[grid[0,0], grid[0,1], grid[0,2]], False)

# ---------------------------------------------------------------------------
# Run for each map
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    for map_name in ['map0', 'map1', 'map2']:
        test_isOccupied(map_name)