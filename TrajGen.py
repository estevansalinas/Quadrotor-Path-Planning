import numpy as np

class TrajGen:
    """
    Use 3rd-order polynomials to fit selected waypoints from the path planner, and generate smooth trajectories to output to the trajectory tracker in Project 2.
    """

    def __init__(self, world, path, planner):
        """
        Parameters:
            world,   World object
            path,    path, ndarray (N, 3) — metric waypoints from start to goal,
            planner,  PlanPath object — used for collision checking
        """
    
        # save the world information
        self.world   = world
        # save the complete path from planner
        self.path    = path
        # save the planner for collision checking
        self.planner = planner
        # Start and goal
        self.start   = np.array(self.world.world['start'])              # shape (3,)
        self.goal    = np.array(self.world.world['goal'])               # shape (3,)

    def prune_waypoints(self):
        """
        Line-of-Sight Pruning: Walk the path and check if you can skip intermediate
        waypoints by drawing a straight line between non-adjacent nodes.
        Parameters:
            path,    ndarray (N, 3) — metric waypoints from start to goal
        Returns:
            pruned,  ndarray (M, 3) — pruned waypoints with redundant
                    intermediate nodes removed, M <= N
        """
        path = self.path
        
        if len(path) <= 2:
            return path

        pruned  = [path[0]]  # add start 
        
        ###################STUDENT CODE ############################

        i = 0 #current waypoint index
        while i < len(path) -1:
            j = i+1
            while j < len(path)-1:
                if self.planner.path_collides(path[i], path[j+1]):
                    break
                j += 1
            pruned.append(path[j])
            i = j #move current index to j
        ###################END OF STUDENT CODE ######################

        return np.array(pruned)
   
    def generate_polynomials(self, pruned, pruned_vel):
        """
        Build and solve the cubic spline system A @ c = b for each axis (x, y, z).
        Each segment i uses: x_i(dt) = c_i,0 + c_i,1*dt + c_i,2*dt^2 + c_i,3*dt^3
        Parameters:
            pruned,      ndarray (m+1, 3) — pruned waypoints, positions p_0 ... p_m
            pruned_vel,  ndarray (m+1, 3) — velocities v_0 ... v_m at each waypoint
        Returns:
            coeffs,      ndarray (m, 4, 3) — polynomial coefficients per segment per axis
                        coeffs[i, :, axis] = [c_i,0  c_i,1  c_i,2  c_i,3]
            T,           ndarray (m,) — duration of each segment
        """

        ###################STUDENT CODE ############################
        m = len(pruned)-1

        T = np.zeros(m)
        for i in range(m):
            T[i] = max(np.linalg.norm(pruned[i+1]-pruned[1]), 0.5)*2

        coeffs = np.zeros((m, 4, 3))

        #solve each axis independentely
        for axis in range(3):
            A = np.zeros((4*m, 4*m))
            b = np.zeros(4*m)

            row = 0
            for i in range(m):
                Ti = T[i]
                col = i*4 #each segment has 4 coefficients

                 # condition 1: x_i(0) = p_i  (start of segment)
                A[row, col:col+4] = [1, 0, 0, 0]
                b[row] = pruned[i, axis]
                row += 1
            
                # condition 2: x_i(T_i) = p_{i+1}  (end of segment)
                A[row, col:col+4] = [1, Ti, Ti**2, Ti**3]
                b[row] = pruned[i+1, axis]
                row += 1
            
                # condition 3: x_i_dot(0) = v_i  (velocity at start)
                A[row, col:col+4] = [0, 1, 0, 0]
                b[row] = pruned_vel[i, axis]
                row += 1
            
                # condition 4: x_i_dot(T_i) = v_{i+1}  (velocity at end)
                A[row, col:col+4] = [0, 1, 2*Ti, 3*Ti**2]
                b[row] = pruned_vel[i+1, axis]
                row += 1
        
            # solve A @ c = b for this axis
            c = np.linalg.solve(A, b)
        
                # store coefficients for each segment
            for i in range(m):
                coeffs[i, :, axis] = c[i*4:(i+1)*4]
    


        ###################END OF STUDENT CODE ######################

        return coeffs, T
    

    