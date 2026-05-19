import numpy as np

class CubicPolyTraj(object):
    """
    Multi-segment 3rd-order polynomial trajectory.
    Each segment i is a cubic polynomial in local time dt = t - t_i:
        x_i(dt) = c_i,0 + c_i,1*dt + c_i,2*dt^2 + c_i,3*dt^3
    """

    def __init__(self, coeffs, T):
        """
        Constructor for the CubicPolyTraj object. A fresh trajectory
        object will be constructed before each mission.
        Parameters:
            coeffs,  ndarray (m, 4, 3) — polynomial coefficients per segment per axis
                     coeffs[i, k, axis] = c_{i,k} for segment i, power k, axis (x/y/z)
            T,       ndarray (m,) — duration of each segment in seconds
        """
        # Polynomial coefficients for all segments and axes
        self.coeffs = coeffs
        # Duration of each segment
        self.T      = T

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.
        Parameters:
            t,   float — current time, s
        Returns:
            flat_output, dict with keys:
                x,        ndarray (3,) — position, m
                x_dot,    ndarray (3,) — velocity, m/s
                x_ddot,   ndarray (3,) — acceleration, m/s^2
                x_dddot,  ndarray (3,) — jerk, m/s^3       (zero for cubic)
                x_ddddot, ndarray (3,) — snap, m/s^4       (zero for cubic)
                yaw,      float — yaw angle, rad            (zero, no yaw tracking)
                yaw_dot,  float — yaw rate, rad/s           (zero, no yaw tracking)
        """
        # Initialize outputs to zero
        x    = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw      = 0
        yaw_dot  = 0

        ###################STUDENT CODE ############################


    
        # find which segment we are in
        t_total = np.sum(self.T)

        # if past the end, hold final position
        if t >= t_total:
            i = len(self.T) - 1  # last segment
            dt = self.T[i]       # end of last segment
        else:
        # walk through segments to find which one t falls in
            i = 0
            t_remaining = t
            while i < len(self.T) - 1 and t_remaining >= self.T[i]:
                t_remaining -= self.T[i]
                i += 1
            dt = t_remaining

        # get coefficients for this segment
        c = self.coeffs[i]  # shape (4, 3)

        # evaluate polynomial for each axis
        # position
        x = c[0] + c[1]*dt + c[2]*dt**2 + c[3]*dt**3

        # velocity
        x_dot = c[1] + 2*c[2]*dt + 3*c[3]*dt**2

        # acceleration
        x_ddot = 2*c[2] + 6*c[3]*dt

        ###################END OF STUDENT CODE ######################
        flat_output = {
            'x':        x,
            'x_dot':    x_dot,
            'x_ddot':   x_ddot,
            'x_dddot':  x_dddot,
            'x_ddddot': x_ddddot,
            'yaw':      yaw,
            'yaw_dot':  yaw_dot
        }

        return flat_output