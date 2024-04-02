

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        # TODO: get the ground truth position by listening to the transformation between /map and /base_link
        # initialize particles by sampling around this area
        # params.yaml provides N as 200, and recommended numpy
        # self.previous_prediction = 

        pass

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO: add the odometry to each particle in their frame
        # TODO: add Gaussian noise with np.random

        # for each particle [x_i y_i theta_i]: 
        # convert [x_i+dx y_i+dy] from the theta_i frame to the theta_i+dtheta frame
        
        # store this prediction
        raise NotImplementedError

        ####################################
