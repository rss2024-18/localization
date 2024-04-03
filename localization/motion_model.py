import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.noise_translational = 0.07  # Standard deviation for translational noise
        self.noise_rotational = 0.74  # Standard deviation for rotational noise
        ####################################
    

    def add_odometry_noise(self, odometry):
        # Add noise to odometry data
        noisy_odometry = odometry.copy()
        noisy_odometry[0] += np.random.normal(0, self.noise_translational)  # Add translational noise
        noisy_odometry[1] += np.random.normal(0, self.noise_translational)
        noisy_odometry[2] += np.random.normal(0, self.noise_rotational)  # Add rotational noise
        return noisy_odometry

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
        # TODO
        # Apply motion model with noise
        noisy_odometry = self.add_odometry_noise(odometry)

        # Update particle positions based on noisy odometry
        for particle in particles:
            theta = particle[2]
            dx = noisy_odometry[0] * np.cos(theta) - noisy_odometry[1] * np.sin(theta)
            dy = noisy_odometry[0] * np.sin(theta) + noisy_odometry[1] * np.cos(theta)
            dtheta = noisy_odometry[2]

            particle[0] += dx
            particle[1] += dy
            particle[2] += dtheta

        return particles
        ####################################
