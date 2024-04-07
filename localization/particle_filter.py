from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion, TransformStamped, PoseWithCovarianceStamped, PointStamped, PoseArray
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
import rclpy
import numpy as np
from sensor_msgs.msg import LaserScan
import threading


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "particle_filter_frame")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter("num_particles", 1000) #check that this value is being imported correctly

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.pose_sub = self.create_subscription(PointStamped, "/clicked_point", self.pose_callback, 10)

        #self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)
        self.tf_pub = self.create_publisher(TransformStamped, "/tf", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        self.particles = None
        self.initialized = False

        self.particle_lock = threading.Lock()

    def motion_update(self, odometry):
        if self.particles is not None:
            self.particles = self.motion_model.evaluate(self.particles, odometry)

    def sensor_update(self, scan):
        if self.particles is not None:
            ## TODO downsample scan first 
            ## right now this works with no downsampling in the sim because the laser scans 
            ## have 100 beams anyways

            weights = self.sensor_model.evaluate(self.particles, scan)
            self.resample_particles(weights)

    def resample_particles(self, weights):
        ## TODO debug np.random.choice since our p is not designed to sum to 1
        ## can normalize? added below for now
        # weights *= 1/np.sum(weights)
        # indices = np.random.choice(len(self.particles), len(self.particles), p=weights)
        # self.particles = self.particles[indices]
        num_samples = len(self.particles)
        selected_indices = np.random.choice(range(len(self.particles)), size=num_samples, p=weights, replace=True)
        self.particles = self.particles[selected_indices]


    def odom_callback(self, msg):
        self.particle_lock.acquire()
        if not self.initialized:
            self.particle_lock.release()
            return
            # pose = msg.pose.pose
            # self.initialize_particles(pose)
            # self.initialized = True
        # Only use the twist component of the odometry message
        odometry = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        self.get_logger().info("odometry"+str(odometry))
        
        #get time
        header_timestamp = msg.header.stamp
        timestamp_sec = header_timestamp.sec
        timestamp_nanosec = header_timestamp.nanosec
        timestamp_seconds = timestamp_sec + timestamp_nanosec / 1e9
    
        updated_particles = self.motion_model.evaluate(self.particles, odometry, timestamp_seconds)
        self.particles = updated_particles
        self.particle_lock.release()
        self.publish_pose()

    def laser_callback(self, msg):
        self.particle_lock.acquire()
        if self.particles is not None:
            scan = [z for z in msg.ranges if z != float('inf')]
            self.sensor_update(scan)
        self.particle_lock.release()

    def pose_callback(self, msg):
        if not self.initialized:
            x = msg.point.x
            y = msg.point.y
            self.initialize_particles(x, y)
            self.initialized = True

    def initialize_particles(self, x, y):
        ####Get From params instead !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        #x, y, theta = pose.position.x, pose.position.y, self.quaternion_to_yaw(pose.orientation)
        self.particles = np.array([[x, y, 0]] * num_particles)
        #todo!!!!! Figure out how to make this scaled wrt the map we are given !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for particle in range(num_particles):
            part = np.random.rand(3) - 0.5  # Generate random offsets between -0.5 and 0.5
            
            # Scale the offsets by 1 meter and add to the original (x, y) position
            self.particles[particle, 0] = x + part[0] * 10.0
            self.particles[particle, 1] = y + part[1] * 10.0
            
            # Generate random orientation in the range [0, 2*pi]
            self.particles[particle, 2] = np.random.uniform(0, 2*np.pi)

    def quaternion_to_yaw(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        scipy_quaternion = (x, y, z, w)
        r = Rotation.from_quat(scipy_quaternion)
        euler_angles = r.as_euler('xyz')
        return euler_angles[2]

    def find_cluster(self):
        db = DBSCAN(eps=0.5).fit(self.particles[:,:2])
        labels = db.labels_
        unique_labels = set(labels)
        largest_cluster_label = max(unique_labels, key=list(labels).count)
        largest_cluster = self.particles[labels == largest_cluster_label]
        return largest_cluster
    
    def mean_angle(self, angles):
        # Convert angles to unit vectors
        unit_vectors = np.column_stack((np.cos(angles), np.sin(angles)))
        # Take the mean of unit vectors
        mean_vector = np.mean(unit_vectors, axis=0)
        # Convert mean vector to angle
        mean_angle = np.arctan2(mean_vector[1], mean_vector[0])
        return mean_angle
    
    def find_average_pos(self):
        largest_cluster = self.find_cluster()
        avgx = np.average(largest_cluster[0])
        avgy = np.average(largest_cluster[1])
        thetas = largest_cluster[2]
        avgtheta = self.mean_angle(thetas)
        return (avgx, avgy, avgtheta)

    def publish_pose(self):
        if self.particles is not None:
            avg_pose = self.find_average_pos()
            pose = Pose()
            pose.position.x, pose.position.y = avg_pose[0], avg_pose[1]
            r = Rotation.from_euler('xyz', [0, 0, avg_pose[2]])
            x, y, z, w = r.as_quat()[0], r.as_quat()[1], r.as_quat()[2], r.as_quat()[3]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w
            self.publish_transform(pose)
            
    def publish_particles(self):
        particle_array_msg = PoseArray()
        particle_array_msg.header.stamp = self.get_clock().now().to_msg()
        particle_array_msg.header.frame_id = "map"

        for particle in self.particles: 
            particle_pose = Pose()
            particle_pose.position.x = particle[0]
            particle_pose.position.y = particle[1]
            particle_pose.position.z = 0.1
            r = Rotation.from_euler('xyz', [0, 0, particle[2]])
            quat = r.as_quat()
            particle_pose.orientation.x = quat[0]
            particle_pose.orientation.y = quat[1]
            particle_pose.orientation.z = quat[2]
            particle_pose.orientation.w = quat[3]
            particle_array_msg.poses.append(particle_pose)
        self.particles_pub.publish(particle_array_msg)

    
    def publish_transform(self, pose):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = self.particle_filter_frame
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = pose.orientation
        self.tf_pub.publish(transform)
        #self.particles_pub.publish(self.particles) #fix this
        self.publish_particles()

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
