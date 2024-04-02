import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        self.N = 200  # Number of particles
        self.particles = np.zeros((self.N, 3))  # Initialize particles
        self.alpha = [0.74, 0.07, 0.07, 0.12]  # Noise parameters
        self.previous_odometry = np.array([0, 0, 0])  # Initialize with zeros or the first odometry reading

        # TODO
        # Do any precomputation for the motion
        # model here.

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
        d_rot1 = np.arctan2(odometry[1], odometry[0]) - self.previous_odometry[2]
        d_trans = np.sqrt(odometry[0]**2 + odometry[1]**2)
        d_rot2 = odometry[2] - self.previous_odometry[2] - d_rot1

        for i in range(self.N):
            # Add odometry to each particle
            x, y, theta = particles[i]
            
            # Sample noise for each motion component
            noisy_d_rot1 = d_rot1 - np.random.normal(0, np.sqrt(self.alpha[0] * d_rot1**2 + self.alpha[1] * d_trans**2))
            noisy_d_trans = d_trans - np.random.normal(0, np.sqrt(self.alpha[2] * d_trans**2 + self.alpha[3] * (d_rot1 + d_rot2)**2))
            noisy_d_rot2 = d_rot2 - np.random.normal(0, np.sqrt(self.alpha[0] * d_rot2**2 + self.alpha[1] * d_trans**2))

            # Apply motion model to update particle position
            particles[i][0] = x + noisy_d_trans * np.cos(theta + noisy_d_rot1)
            particles[i][1] = y + noisy_d_trans * np.sin(theta + noisy_d_rot1)
            particles[i][2] = theta + noisy_d_rot1 + noisy_d_rot2

        self.previous_odometry = odometry.copy()  # Update the previous odometry with the current odometry

        return particles

        ####################################