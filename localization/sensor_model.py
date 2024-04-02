import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        # Discretization step for range values
        max_range = self.table_width - 1  # Assuming max range in "pixels" is table_width - 1
        delta = max_range / (self.table_width - 1)

        # Initialize the sensor model table with zeros
        self.sensor_model_table = np.zeros((self.table_width, self.table_width))

        # Iterate over true distances (i) and observed distances (j)
        for i in range(self.table_width):
            true_distance = i * delta  # Convert index to distance

            # Compute hit probabilities
            hit_probabilities = np.exp(-0.5 * ((np.arange(self.table_width) * delta - true_distance) ** 2) / (self.sigma_hit ** 2))
            hit_probabilities /= (self.sigma_hit * np.sqrt(2 * np.pi))  # Normalize the Gaussian

            # Compute short probabilities
            short_probabilities = np.zeros(self.table_width)
            short_probabilities[:i] = 2 * (1 / true_distance) * (1 - np.arange(i) * delta / true_distance)
            
            # Compute max probabilities
            max_probabilities = np.zeros(self.table_width)
            max_probabilities[-1] = 1  # All probability mass at the max range

            # Compute random probabilities
            random_probabilities = np.ones(self.table_width) * (1 / max_range)

            # Combine probabilities
            combined_probabilities = (self.alpha_hit * hit_probabilities +
                                    self.alpha_short * short_probabilities +
                                    self.alpha_max * max_probabilities +
                                    self.alpha_rand * random_probabilities)

            # Normalize combined probabilities to sum to 1
            combined_probabilities /= np.sum(combined_probabilities)

            # Assign to the sensor model table
            self.sensor_model_table[i, :] = combined_probabilities

    print("Sensor model table precomputed")





    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """


        if not self.map_set:
            return np.ones(len(particles)) / len(particles)  # Uniform distribution if map is not set

        # Evaluate sensor model for each particle
        weights = []
        scans = self.scan_sim.scan(particles)
        for particle_scan in scans:
            weight = 1.0
            for observed_range, expected_range in zip(observation, particle_scan):
                if observed_range < self.table_width:
                    probs = self.sensor_model_table[int(observed_range)]
                    weight *= probs[int(expected_range)]
            weights.append(weight)
        
        # Normalize weights
        weights /= np.sum(weights)

        return weights

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
