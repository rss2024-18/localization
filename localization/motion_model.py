import numpy as np
from builtin_interfaces.msg import Duration

class MotionModel:

    def __init__(self, node):
        self.node = node
        self.initialized = False
        self.last_time = None
        # pass
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.noise_translational = 0.05  # 0.07 # Standard deviation for translational noise 
        self.noise_rotational = 0.1 # 0.74  # Standard deviation for rotational noise
        ####################################
    

    def add_odometry_noise(self, odometry):
        # Add noise to odometry data
        noisy_odometry = odometry.copy()
        noisy_odometry[0] += np.random.normal(0, self.noise_translational)  # Add translational noise
        noisy_odometry[1] += np.random.normal(0, self.noise_translational)
        noisy_odometry[2] += np.random.normal(0, self.noise_rotational)  # Add rotational noise
        return noisy_odometry

    def evaluate(self, particles, odometry, current_time):
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
        noisy_odometry = [0.0, 0.0, 0.0]
        noisy_odometry[0] = odometry.copy()[0]
        noisy_odometry[1] = odometry.copy()[1]
        noisy_odometry[2] = odometry.copy()[2]
        new_particles=particles.copy()
        if not self.initialized:
            self.last_time = current_time
            self.initialized = True

        delta_time = current_time - self.last_time
        #delta_time =  duration.seconds() # + duration.nanoseconds()
        
        #car odometry 
        theta_change = noisy_odometry[2]*delta_time
        x_disp = noisy_odometry[0]*np.cos(theta_change)*delta_time
        y_disp = noisy_odometry[0]*np.sin(theta_change)*delta_time


        ind = -1
        for particle in particles:
            ind = ind + 1
            theta = particle[2]

            temp_odometry = noisy_odometry
            dx = temp_odometry[0]*np.cos(theta_change + theta)*delta_time
            dy = temp_odometry[0]*np.sin(theta_change + theta)*delta_time
            dtheta = temp_odometry[2] * delta_time
            temp_odometry = self.add_odometry_noise([dx, dy, dtheta])

            # dx = noisy_odometry[0]* delta_time * np.cos(theta) - noisy_odometry[1] * delta_time * np.sin(theta)
            # dy = noisy_odometry[0]*delta_time * np.sin(theta) + noisy_odometry[1] * delta_time * np.cos(theta)
            # dtheta = noisy_odometry[2] * delta_time

            particle[0] += temp_odometry[0]
            particle[1] += temp_odometry[1]
            particle[2] += temp_odometry[2]
            new_particles[ind] = particle
    
        self.last_time = current_time

        return new_particles
        ####################################
