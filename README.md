# Quadrotor Path Planning

3D path planning for a quadrotor UAV using Dijkstra and A* search algorithms on a discretized occupancy grid, with cubic polynomial trajectory generation. Implemented in Python as part of an Aerial Robots and Visual Navigation course at the University of Houston–Clear Lake.

## Overview

The planner discretizes a 3D world into a voxel grid, inflates obstacles by a configurable margin, and searches for a collision-free optimal path from start to goal. The resulting waypoints are pruned and connected with smooth cubic polynomial trajectories, then tracked by a PD controller in simulation.

## Pipeline

```
World (JSON) → Occupancy Grid → Path Search (Dijkstra / A*) → Waypoint Pruning → Cubic Polynomial Trajectory → PD Controller → Simulation
```
## Algorithms

| Algorithm | Description |
|-----------|-------------|
| Dijkstra | Uniform-cost search, guarantees shortest path |
| A* | Heuristic-guided search (Euclidean distance), faster than Dijkstra |

## File Structure

| File | Description |
|------|-------------|
| `PlanPath.py` | Occupancy grid construction, Dijkstra, and A* implementation |
| `TrajGen.py` | Waypoint pruning and polynomial coefficient generation |
| `CubicPolyTraj.py` | Cubic polynomial trajectory evaluation |
| `main.py` | Simulation entry point with plotting and animation |
| `maps/` | JSON world files with obstacles, start, and goal |
| `tests/` | Unit tests for planning and trajectory modules |

## Tools & Dependencies

- Python, NumPy, SciPy, Matplotlib
- Custom quadrotor flight simulator (`flightsim`)
- PD controller for trajectory tracking
