from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion, TransformStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from scikit_learn.cluster import DBSCAN
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

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1)

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
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
            self.get_logger().info(str(len(scan)))
            weights = self.sensor_model.evaluate(self.particles, scan)
            self.resample_particles(weights)

    def resample_particles(self, weights):
        ## TODO debug np.random.choice since our p is not designed to sum to 1
        ## can normalize? added below for now
        weights *= 1/np.sum(weights)
        indices = np.random.choice(len(self.particles), len(self.particles), p=weights)
        self.particles = self.particles[indices]

    def odom_callback(self, msg):
        self.particle_lock.acquire()
        # Only use the twist component of the odometry message
        odometry = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        updated_particles = self.motion_model.evaluate(self.particles, odometry)
        self.particles = updated_particles
        self.particle_lock.release()

    def laser_callback(self, msg):
        self.particle_lock.acquire()
        if self.particles is not None:
            scan = [z for z in msg.ranges if z != float('inf')]
            self.sensor_update(scan)
        self.particle_lock.release()

    def pose_callback(self, msg):
        if not self.initialized:
            pose = msg.pose.pose
            self.initialize_particles(pose)
            self.initialized = True

    def initialize_particles(self, pose):
        ####Get From params instead !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        num_particles = 1000
        x, y, theta = pose.position.x, pose.position.y, self.quaternion_to_yaw(pose.orientation)
        self.particles = np.array([[x, y, theta]] * num_particles)
        #todo!!!!! Figure out how to make this scaled wrt the map we are given !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for particle in range(len(self.particles[1])):
            part = np.random.rand(1,3) - 0.5 
            part[0,1] *= 10
            part[2] *= 2*np.pi
            self.particles[particle] += part 

    def quaternion_to_yaw(self, quaternion):
        r = Rotation.from_quat(quaternion)
        euler_angles = r.as_euler('xyz')
        return euler_angles[2]

    def find_cluster(self):
        db = DBSCAN(eps=0.5).fit(self.particles[:,:2])
        labels = db.labels_
        unique_labels = set(labels)
        largest_cluster_label = max(unique_labels, key=list(labels).count)
        largest_cluster = self.particles[labels == largest_cluster_label]
        return largest_cluster
    
    def mean_angle(angles):
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
            pose.position.x, pose.position.y = avg_pose(0), avg_pose(1)
            r = Rotation.from_euler('xyz', [0, 0, avg_pose(2)])
            pose.orientation = r.as_quat()
            self.publish_transform(pose)

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

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
