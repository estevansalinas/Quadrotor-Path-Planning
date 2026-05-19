from pathlib import Path
import inspect
import numpy as np
from flightsim.world import World
from project3.PlanPath import PlanPath

# ---------------------------------------------------------------------------
# Load the map0, do not use other maps for this test.
# ---------------------------------------------------------------------------
filename = 'map0.json'
file = Path(inspect.getsourcefile(lambda: 0)).parent.resolve() / '..' / 'maps' / filename
my_world = World.from_file(file)

print(f"Testing map: {filename}")

# ---------------------------------------------------------------------------
# Resolution parameters — change here only
# ---------------------------------------------------------------------------
res_xy = 0.2
res_z  = 0.2
margin = 0.25

pp = PlanPath(my_world, res_xy=res_xy, res_z=res_z, margin=margin)

# Derived grid parameters
bounds  = my_world.world['bounds']['extents']   # [xmin, xmax, ymin, ymax, zmin, zmax]
xmin, xmax = bounds[0], bounds[1]
ymin, ymax = bounds[2], bounds[3]
zmin, zmax = bounds[4], bounds[5]
res_xyz = np.array([res_xy, res_xy, res_z])

def metric_to_grid_expected(pt):
    """Compute expected grid index for a single metric point."""
    origin = np.array([xmin, ymin, zmin])
    return np.floor((np.array(pt) - origin) / res_xyz).astype(int)

def grid_to_metric_expected(idx):
    """Compute expected cell centre for a grid index."""
    origin = np.array([xmin, ymin, zmin])
    return origin + (np.array(idx) + 0.5) * res_xyz

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def check(label, got, expected, tol=1e-6):
    ok = np.allclose(got, expected, atol=tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if not ok:
        print(f"       expected: {expected}")
        print(f"       got     : {got}")

# ---------------------------------------------------------------------------
# 1. metric_to_grid  — returns (N, 3)
# ---------------------------------------------------------------------------

# 1a. Start point [-4, -4, 1.5]
pt_start = [-4.0, -4.0, 1.5]
exp      = metric_to_grid_expected(pt_start)
grid     = pp.metric_to_grid(np.array([pt_start]))
check(f"metric_to_grid: start {pt_start} → row={exp[0]}", grid[0, 0], exp[0])
check(f"metric_to_grid: start {pt_start} → col={exp[1]}", grid[0, 1], exp[1])
check(f"metric_to_grid: start {pt_start} → lyr={exp[2]}", grid[0, 2], exp[2])

# 1b. Goal point [4, 4, 1.5]
pt_goal = [4.0, 4.0, 1.5]
exp     = metric_to_grid_expected(pt_goal)
grid    = pp.metric_to_grid(np.array([pt_goal]))
check(f"metric_to_grid: goal {pt_goal} → row={exp[0]}", grid[0, 0], exp[0])
check(f"metric_to_grid: goal {pt_goal} → col={exp[1]}", grid[0, 1], exp[1])
check(f"metric_to_grid: goal {pt_goal} → lyr={exp[2]}", grid[0, 2], exp[2])

# 1c. Origin corner
pt_origin = [xmin, ymin, zmin]
grid      = pp.metric_to_grid(np.array([pt_origin]))
check(f"metric_to_grid: origin {pt_origin} → (0,0,0)", grid[0], [0, 0, 0])

# 1d. Batch: start and goal together
pts  = np.array([pt_start, pt_goal])
grid = pp.metric_to_grid(pts)
exp_start = metric_to_grid_expected(pt_start)
exp_goal  = metric_to_grid_expected(pt_goal)
check("metric_to_grid: batch rows", grid[:, 0], [exp_start[0], exp_goal[0]])
check("metric_to_grid: batch cols", grid[:, 1], [exp_start[1], exp_goal[1]])
check("metric_to_grid: batch lyrs", grid[:, 2], [exp_start[2], exp_goal[2]])

# ---------------------------------------------------------------------------
# 2. grid_to_metric  — takes (N, 3), returns (N, 3)
# ---------------------------------------------------------------------------

# 2a. Cell (0, 0, 0) centre
exp = grid_to_metric_expected([0, 0, 0])
pts = pp.grid_to_metric(np.array([[0, 0, 0]]))
check(f"grid_to_metric: (0,0,0) centre → {exp}", pts[0], exp)

# 2b. Start cell centre
idx_start = metric_to_grid_expected(pt_start)
exp       = grid_to_metric_expected(idx_start)
pts       = pp.grid_to_metric(np.array([idx_start]))
check(f"grid_to_metric: start cell {idx_start} → {exp}", pts[0], exp)

# 2c. Round-trip: metric → grid → metric stays within half a cell
original  = np.array([pt_start])
grid      = pp.metric_to_grid(original)
recovered = pp.grid_to_metric(grid)
half_cell = res_xyz / 2.0
within    = np.all(np.abs(recovered[0] - original[0]) <= half_cell + 1e-9)
print(f"[{'PASS' if within else 'FAIL'}] grid round-trip: recovered within half-cell of original")

# ---------------------------------------------------------------------------
# 3. grid_to_flat / flat_to_grid
# ---------------------------------------------------------------------------
nz = pp.grid_shape[2]   # number of z layers  = floor((zmax-zmin)/res_z)
ny = pp.grid_shape[1]   # number of y columns = floor((ymax-ymin)/res_xy)

# 3a. (0, 0, 0) → 0
flat = pp.grid_to_flat(np.array([[0, 0, 0]]))
check("grid_to_flat: (0,0,0) → 0", flat[0], 0)

# 3b. (0, 1, 0) → nz  (next column: skip one full z-strip)
flat = pp.grid_to_flat(np.array([[0, 1, 0]]))
check(f"grid_to_flat: (0,1,0) → {nz}", flat[0], nz)

# 3c. (1, 0, 0) → ny*nz  (next row: skip one full col*lyr slab)
flat = pp.grid_to_flat(np.array([[1, 0, 0]]))
check(f"grid_to_flat: (1,0,0) → {ny*nz}", flat[0], ny * nz)

# 3d. Inverse: flat_to_grid
idx      = list(idx_start)
flat_val = pp.grid_to_flat(np.array([idx]))
grid     = pp.flat_to_grid(flat_val)
check(f"flat_to_grid: recovers {idx} row", grid[0, 0], idx[0])
check(f"flat_to_grid: recovers {idx} col", grid[0, 1], idx[1])
check(f"flat_to_grid: recovers {idx} lyr", grid[0, 2], idx[2])

# 3e. Start and goal flat indices differ
flat_start = pp.grid_to_flat(np.array([metric_to_grid_expected(pt_start)]))
flat_goal  = pp.grid_to_flat(np.array([metric_to_grid_expected(pt_goal)]))
check("flat: start != goal", flat_start[0] != flat_goal[0], True)

# ---------------------------------------------------------------------------
# 4. metric_to_flat / flat_to_metric  (composed helpers)
# ---------------------------------------------------------------------------

# 4a. metric_to_flat: start
flat          = pp.metric_to_flat(np.array([pt_start]))
expected_flat = pp.grid_to_flat(np.array([metric_to_grid_expected(pt_start)]))
check("metric_to_flat: start", flat[0], expected_flat[0])

# 4b. flat_to_metric: round-trip stays within half a cell
recovered = pp.flat_to_metric(flat)
within    = np.all(np.abs(recovered[0] - np.array(pt_start)) <= half_cell + 1e-9)
print(f"[{'PASS' if within else 'FAIL'}] flat_to_metric round-trip: within half-cell")

# 4c. metric_to_flat: goal
flat_g          = pp.metric_to_flat(np.array([pt_goal]))
expected_flat_g = pp.grid_to_flat(np.array([metric_to_grid_expected(pt_goal)]))
check("metric_to_flat: goal", flat_g[0], expected_flat_g[0])

# ---------------------------------------------------------------------------
# 5. Edge / boundary cases
# ---------------------------------------------------------------------------

# 5a. z layers
pt_z0   = [xmin, ymin, zmin]
pt_ztop = [xmin, ymin, zmax - 1e-9]   # just inside top
exp_z0  = metric_to_grid_expected(pt_z0)[2]
exp_zt  = metric_to_grid_expected(pt_ztop)[2]
grid    = pp.metric_to_grid(np.array([pt_z0, pt_ztop]))
check(f"metric_to_grid: z={zmin} → lyr={exp_z0}", grid[0, 2], exp_z0)
check(f"metric_to_grid: z≈{zmax} → lyr={exp_zt}", grid[1, 2], exp_zt)

# 5b. Sample points info
for label, pt in [("wall-1", [-2.0, -1.95, 1.5]), ("wall-2", [2.0, 1.95, 1.5])]:
    grid = pp.metric_to_grid(np.array([pt]))
    print(f"[INFO] {label} sample cell: row={grid[0,0]}, col={grid[0,1]}, lyr={grid[0,2]}")

print("\nDone.")