import numpy as np
import heapq

class PlanPath:
    """
    Discretizes 3D space and applies Dijkstra or A* search to find collision-free optimal paths.
    """

    def __init__(self, world, res_xy=0.2, res_z=0.2, margin=0.25):
        """
        Parameters:
            world,   World object
            res_xy,  float — cell size in x and y (meters)
            res_z,   float — cell size in z (meters)
            margin,  float — inflation around obstacles (meters)
        """
        self.margin  = margin
        self.res_xyz = np.array([res_xy, res_xy, res_z], dtype=np.float64)
        
        # Bounds
        self.bounds  = np.array(world.world['bounds']['extents'])  # [xmin,xmax,ymin,ymax,zmin,zmax]

        # Obstacles
        self.blocks  = np.array([b['extents'] for b in world.world.get('blocks', [])])  # shape (N, 6) : [xmin,xmax,ymin,ymax,zmin,zmax]

        # Start and goal
        self.start   = np.array(world.world['start'])              # shape (3,)
        self.goal    = np.array(world.world['goal'])               # shape (3,)

        self.origin = np.array([self.bounds[0],   # xmin
                         self.bounds[2],   # ymin
                         self.bounds[4]])  # zmin
        
        self.grid_shape   = tuple(np.ceil((self.bounds[[1, 3, 5]] - self.bounds[[0, 2, 4]]) / self.res_xyz).astype(int)) # Grid shape: (nx, ny, nz) — number of cells along each axis.
        self.num_cells = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2] # num_cells = nx * ny * nz 
        
        self.isOccupied = np.zeros(self.grid_shape, dtype=bool)  # (nx, ny, nz) — False = free, True = occupied 


        self.isOccupied_Init() # Build isOccupied array

    # ---------------------------------------------------------------------------
    # Coordinate systems used in path planning
    # ---------------------------------------------------------------------------
    #
    #  METRIC  points (x, y, z) — real-world continuous coordinates in meters
    #          e.g. (-4.0, -4.0, 1.5). Can have fractional values.
    #          Origin is at the world boundary minimum corner.
    #
    #  GRID    grid (row, col, lyr) — discrete 3D indices into the occupancy grid
    #          e.g. (0, 0, 2). Always non-negative integers.
    #          Each cell represents a volume of res_xy × res_xy × res_z meters.
    #
    #  FLAT    flat_idx (ind) — single integer index into the flattened occupancy grid
    #          Useful for graph-based planners (A*, Dijkstra) where nodes
    #          are stored as 1D arrays.

    def metric_to_grid(self, points):
        """
        Convert metric coordinates to grid subscripts.

        Parameters:
            points, ndarray (N, 3) — metric (x, y, z) positions

        Returns:
            grid — ndarray (N, 3) — grid indices

        Mapping:
            metric ──metric_to_grid──► (row, col, lyr)
        """
        ###################STUDENT CODE ############################

        
        grid = np.floor((points - self.origin)/self.res_xyz).astype(int) #formula: grid = floor((points - origin)/res)

        ###################END OF STUDENT CODE ######################
        return grid

    def grid_to_metric(self, grid):
        """
        Convert grid subscripts to metric coordinates.

        Parameters:
            grid, ndarray (N, 3) — grid indices

        Returns:
            points, ndarray (N, 3) — metric (x, y, z) centre of each cell

        Mapping:
            grid (row, col, lyr) ── grid_to_metric ──► metric
        """
        ###################STUDENT CODE #############################


        points = self.origin +(grid + 0.5)*self.res_xyz #formula: points= origin + (grid + 0.5) * res

        ###################END OF STUDENT CODE ######################
        return points

    def grid_to_flat(self, grid):
        """
        Convert grid subscripts to flat 1D index.

        Parameters:
            grid  — ndarray (N, 3) — grid indices

        Returns:
            flat_idx, ndarray (N,) — 1D index into occgrid

        Mapping:
            grid ──grid_to_flat──► flat
        """
        ###################STUDENT CODE #############################

        flat_idx = grid[:, 0]*(self.grid_shape[1]*self.grid_shape[2]) + grid[:, 1]*self.grid_shape[2] + grid[:, 2] #formula: flat_idx = row * (ncols * nlayers) + col*nlayers +lyr

        ###################END OF STUDENT CODE ######################
        return flat_idx

    def flat_to_grid(self, flat_idx):
        """
        Convert flat 1D index to grid subscripts.

        Parameters:
            flat_idx, ndarray (N,) or scalar — 1D index into occgrid

        Returns:
            grid — each ndarray (N, 3) — grid indices

        Mapping:
            flat ──flat_to_grid──► (row, col, lyr)
        """
        ###################STUDENT CODE ############################
        
        row = flat_idx // (self.grid_shape[1] * self.grid_shape[2])
        col = (flat_idx%(self.grid_shape[1]*self.grid_shape[2])) //self.grid_shape[2]
        lyr = flat_idx%self.grid_shape[2]
        grid = np.column_stack([row,col,lyr])
        
        #formula: row = floor( flat_idx / (ncols*nlayers) )
                 #col = floor( (flat_idx % (ncols*nlayers)) / nlayers )
                 #lyr = flat_idx % nlayers

        ###################END OF STUDENT CODE ######################
        return grid

    def metric_to_flat(self, points):
        """
        Convert metric coordinates to flat 1D index.

        Parameters:
            points, ndarray (N, 3) — metric (x, y, z) positions

        Returns:
            flat_idx, ndarray (N,) — 1D index into occgrid

        Mapping:
            metric ──metric_to_grid──► (row,col,lyr) ──grid_to_flat──► flat
        """
        ###################STUDENT CODE #############################

        flat_idx = self.grid_to_flat(self.metric_to_grid(points)) #formula: flat_idx = self.grid_to_flat( floor(points - origin / res) )


        ###################END OF STUDENT CODE ######################
        return flat_idx

    def flat_to_metric(self, flat_idx):
        """
        Convert flat 1D index to metric coordinates.

        Parameters:
            flat_idx, ndarray (N,) or scalar — 1D index into occgrid

        Returns:
            points, ndarray (N, 3) — metric (x, y, z) centre of each cell

        Mapping:
            flat ──flat_to_grid──► grid ──grid_to_metric──► metric
        """
        ###################STUDENT CODE #############################

        points = self.grid_to_metric(self.flat_to_grid(flat_idx)) #formula: points = origin + (self.flat_to_grid(flat_idx) + 0.5) × res

        ###################END OF STUDENT CODE ######################
        return points
    

    def isOccupied_Init(self):
        """
        Build isOccupied array by marking all grid cells whose centre
        falls inside any obstacle block (plus margin inflation).

        For each block extents [xmin,xmax,ymin,ymax,zmin,zmax]:
            - inflate by self.margin on all sides
            - convert inflated corners to grid indices
            - mark all cells within that range as True
        """

        for block in self.blocks:
    # loop for every obstacles
            xmin, xmax, ymin, ymax, zmin, zmax = block
            xmin_inf = xmin - self.margin
            xmax_inf = xmax + self.margin
            ymin_inf = ymin - self.margin
            ymax_inf = ymax + self.margin
            zmin_inf = zmin - self.margin
            zmax_inf = zmax + self.margin

    # corners
            corner1 = np.array([[xmin_inf, ymin_inf, zmin_inf]])  # shape (1,3)
            corner2 = np.array([[xmax_inf, ymax_inf, zmax_inf]])  # shape (1,3)

    # convert to grid
            grid1 = self.metric_to_grid(corner1)
            grid2 = self.metric_to_grid(corner2)

    # take actual min and max of the two corners (handles unsorted extents)
            grid_min = np.minimum(grid1, grid2)
            grid_max = np.maximum(grid1, grid2)

    # clamp to keep indices inside grid
            grid_min = np.clip(grid_min, 0, np.array(self.grid_shape) - 1)
            grid_max = np.clip(grid_max, 0, np.array(self.grid_shape) - 1)

            self.isOccupied[grid_min[0,0]:grid_max[0,0]+1,
                    grid_min[0,1]:grid_max[0,1]+1,
                    grid_min[0,2]:grid_max[0,2]+1] = True
        
    def runDijkstra(self):
        """
        Find shortest path from start to goal using Dijkstra's algorithm
        on the 3D occupancy grid.

        Returns:
            path, ndarray (N, 3) — metric waypoints from start to goal,
                                or None if no path exists.
        """

        ###################STUDENT CODE ############################
        deltas = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                for dl in [-1, 0, 1]:
                    if dr == 0 and dc == 0 and dl == 0:
                        continue
                    deltas.append([dr, dc, dl])
        deltas = np.array(deltas) 

        #convert start/goal to flat indices
        start_flat = self.metric_to_flat(self.start.reshape(1,3))[0]
        goal_flat  = self.metric_to_flat(self.goal.reshape(1,3))[0]

        #initialize unknown costs to infinity, previous nodes to -1
        dist    = np.full(self.num_cells, np.inf)
        prev    = np.full(self.num_cells, -1, dtype=int)
        visited = np.zeros(self.num_cells, dtype=bool)

        #cost to reach start is 0
        dist[start_flat] = 0.0
        pq = []
        heapq.heappush(pq, (0.0, start_flat))
        
        #main loop
        while pq:
            cost, current = heapq.heappop(pq) #get cheapest node

            if visited[current]:
                continue                       #skip if already cheapest
            visited[current] = True

            if current == goal_flat:
                break

            #get current cells grid position
            current_grid = self.flat_to_grid(np.array([current]))[0]
            
            for delta in deltas:
                neighbor_grid = current_grid+delta

                #skip if outside bounds 
                r, c, l = neighbor_grid
                if r<0 or c<0 or l<0:
                    continue
                if r >= self.grid_shape[0] or c >= self.grid_shape[1] or l >= self.grid_shape[2]:
                    continue
                
                #skip if occupied
                if self.isOccupied[neighbor_grid[0], neighbor_grid[1], neighbor_grid[2]]:
                    continue

                #get flat index of neighbor
                neighbor_flat = self.grid_to_flat(neighbor_grid.reshape(1,3))[0]

                if visited[neighbor_flat]:      #skip if already processed
                    continue

                #calculate cost
                move_cost = np.linalg.norm(delta*self.res_xyz)
                new_cost  = dist[current] + move_cost

                #update if cheaper path found
                if new_cost < dist[neighbor_flat]:
                    dist[neighbor_flat] = new_cost
                    prev[neighbor_flat] = current
                    heapq.heappush(pq, (new_cost, neighbor_flat))
                
                #reconstruct path by tracing back from goal to start
                #if no path is found:
        if prev[goal_flat] == -1 and start_flat != goal_flat:
            path = None 
        else: 
             waypoints = []
             node = goal_flat
             while node != -1: 
                 waypoints.append(node)
                 node = prev[node]
                    
             waypoints.reverse()

          #convert flat indices to metric coordinates
             waypoints = np.array(waypoints)
             path = self.flat_to_metric(waypoints)

        ###################END OF STUDENT CODE ######################
        return path
        

    def runAstar(self):
        """
        Find shortest path from start to goal using A* algorithm
        on the 3D occupancy grid.

        Returns:
            path, ndarray (N, 3) — metric waypoints from start to goal,
                                or None if no path exists.

        """

        ###################STUDENT CODE ############################
        deltas = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                for dl in [-1, 0, 1]:
                    if dr == 0 and dc == 0 and dl == 0:
                        continue
                    deltas.append([dr, dc, dl])
        deltas = np.array(deltas) 

        #convert start/goal to flat indices
        start_flat = self.metric_to_flat(self.start.reshape(1,3))[0]
        goal_flat  = self.metric_to_flat(self.goal.reshape(1,3))[0]

        #initialize unknown costs to infinity, previous nodes to -1
        dist    = np.full(self.num_cells, np.inf)
        prev    = np.full(self.num_cells, -1, dtype=int)
        visited = np.zeros(self.num_cells, dtype=bool)

        #cost to reach start is 0
        dist[start_flat] = 0.0
        pq = []
        heapq.heappush(pq, (0.0, start_flat))
        
        #main loop
        while pq:
            cost, current = heapq.heappop(pq) #get cheapest node

            if visited[current]:
                continue                       #skip if already cheapest
            visited[current] = True

            if current == goal_flat:
                break

            #get current cells grid position
            current_grid = self.flat_to_grid(np.array([current]))[0]
            
            for delta in deltas:
                neighbor_grid = current_grid+delta

                #skip if outside bounds 
                r, c, l = neighbor_grid
                if r<0 or c<0 or l<0:
                    continue
                if r >= self.grid_shape[0] or c >= self.grid_shape[1] or l >= self.grid_shape[2]:
                    continue
                
                #skip if occupied
                if self.isOccupied[neighbor_grid[0], neighbor_grid[1], neighbor_grid[2]]:
                    continue

                #get flat index of neighbor
                neighbor_flat = self.grid_to_flat(neighbor_grid.reshape(1,3))[0]

                if visited[neighbor_flat]:      #skip if already processed
                    continue

                #calculate cost
                move_cost = np.linalg.norm(delta*self.res_xyz)
                new_cost  = dist[current] + move_cost

                #update if cheaper path found
                if new_cost < dist[neighbor_flat]:
                    dist[neighbor_flat] = new_cost
                    prev[neighbor_flat] = current

                    #calculate heuristic
                    h = np.linalg.norm(self.flat_to_metric(np.array([neighbor_flat])) - self.goal)
                    #push cost+heuristic 
                    heapq.heappush(pq, (new_cost + h, neighbor_flat))
                
                #reconstruct path by tracing back from goal to start
                #if no path is found:
        if prev[goal_flat] == -1 and start_flat != goal_flat:
            path = None 
        else: 
             waypoints = []
             node = goal_flat
             while node != -1: 
                 waypoints.append(node)
                 node = prev[node]
                    
             waypoints.reverse()

          #convert flat indices to metric coordinates
             waypoints = np.array(waypoints)
             path = self.flat_to_metric(waypoints)

        ###################END OF STUDENT CODE ######################
        return path
    
    def path_collides(self, a, b):
        """
        Check if the straight line segment between two metric points passes
        through any occupied cell in the occupancy grid.
        This will be used in waypoint pruning in trajectory generation.
        Parameters:
            a,   ndarray (3,) — start point in metric coordinates
            b,   ndarray (3,) — end point in metric coordinates
        Returns:
            bool — True if the segment collides with an obstacle, False if clear
        """
        # Sample points along the segment at half-cell resolution
        dist    = np.linalg.norm(b - a)
        n_steps = max(int(dist / (min(self.res_xyz) * 0.5)), 2)
        ts      = np.linspace(0, 1, n_steps)          # (n_steps,)
        points  = a + ts[:, None] * (b - a)           # (n_steps, 3)
        grid    = self.metric_to_grid(points)          # (n_steps, 3)
        # Clamp to grid bounds
        row = np.clip(grid[:, 0], 0, self.grid_shape[0] - 1)
        col = np.clip(grid[:, 1], 0, self.grid_shape[1] - 1)
        lyr = np.clip(grid[:, 2], 0, self.grid_shape[2] - 1)
        return bool(np.any(self.isOccupied[row, col, lyr]))