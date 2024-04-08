from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion, TransformStamped
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

import tf2_ros
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
import threading

assert rclpy

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('num_particles', "default")
        self.declare_parameter('num_beams_per_particle', "default")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.tfBuffer = tf2_ros.Buffer()
        self.tfBroadcaster = tf2_ros.TransformBroadcaster(self)
        self.particles_pub = self.create_publisher(PoseArray, "/pf/pose/particles", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().integer_value
        self.particles = None
        self.particle_lock = threading.Lock()

    def pose_callback(self, pose_estimate):
        # respond to an initial pose estimate and initialize points

        self.particle_lock.acquire()
        self.particles = np.empty((self.num_particles, 3))
        self.particle_lock.release()
        mean = np.array([pose_estimate.pose.pose.position.x, pose_estimate.pose.pose.position.y, pose_estimate.pose.pose.position.z])
        use_covariance = False
        if use_covariance:
            x, y, z = np.random.multivariate_normal(mean, np.array(pose_estimate.covariance), self.num_particles)
        else:
            sigma = 0.1
            x = np.random.normal(mean[0], sigma, self.num_particles)
            y = np.random.normal(mean[1], sigma, self.num_particles)

        self.particle_lock.acquire()
        self.particles[:,0] = x
        self.particles[:,1] = y
        self.particles[:,2] = np.random.uniform(0, 2*np.pi, size=self.num_particles)
        self.particle_lock.release()

        self.publish_particles()
        self.publish_pose()

    def laser_callback(self, laser_scan): 
        if self.particles is None:
            return
        
        distances = np.array(laser_scan.ranges)

        # downsample
        downsampled_indices = np.round(np.linspace(0, self.num_beams_per_particle-1, self.num_beams_per_particle)).astype(int)
        downsampled = distances[downsampled_indices]

        # get probabilities from sensor model and resample particles
        probabilities = self.sensor_model.evaluate(self.particles, downsampled)
        probabilities *= 1 / np.sum(probabilities) # normalize for np.random.choice
        resampled_indices = np.random.choice(self.num_particles, size=self.num_particles, p=probabilities)
        self.particle_lock.acquire()
        self.particles = self.particles[resampled_indices]
        self.particle_lock.release()

        self.publish_pose()      
        self.publish_particles()

    def odom_callback(self, odometry):
        if self.particles is None:
            return

        # only use the twist component of the odometry message
        pose = [odometry.twist.twist.linear.x, odometry.twist.twist.linear.y, odometry.twist.twist.angular.z]
        
        # get time
        header_timestamp = odometry.header.stamp
        timestamp_sec = header_timestamp.sec
        timestamp_nanosec = header_timestamp.nanosec
        timestamp_seconds = timestamp_sec + timestamp_nanosec / 1e9
    
        # update particles from odometry
        updated_particles = self.motion_model.evaluate(self.particles, pose, timestamp_seconds)

        self.particle_lock.acquire()
        self.particles = updated_particles
        self.particle_lock.release()

        self.publish_pose()  
        self.publish_particles()

    def publish_pose(self):
        if self.particles is None:
            return
        
        avg_pose = self.find_average_pos()
        pose = Pose()
        pose.position.x, pose.position.y = avg_pose[0], avg_pose[1]
        r = Rotation.from_euler('xyz', [0, 0, avg_pose[2]])
        x, y, z, w = r.as_quat()[0], r.as_quat()[1], r.as_quat()[2], r.as_quat()[3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w

        # publish as odometry
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "/map"
        odom.child_frame_id = self.particle_filter_frame
        odom.pose.pose = pose
        self.odom_pub.publish(odom)

        # send as transform
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "/map"
        tf.child_frame_id = self.particle_filter_frame
        tf.transform.translation.x = pose.position.x
        tf.transform.translation.y = pose.position.y
        tf.transform.rotation = pose.orientation

        self.tfBroadcaster.sendTransform(tf)

    def find_average_pos(self): #can literally get the average of all the particles instead
        # largest_cluster = self.find_cluster()
        # avgx = np.average(largest_cluster[0])
        # avgy = np.average(largest_cluster[1])
        # thetas = largest_cluster[2]
        avgx = np.mean(self.particles[:,0])
        avgy = np.mean(self.particles[:,1])
        avgtheta = self.mean_angle(self.particles[:,2])
        return (avgx, avgy, avgtheta)

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
    
    def publish_particles(self):
        if self.particles is None:
            return

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "/map"

        for particle in self.particles: 
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0.1
            r = Rotation.from_euler('xyz', [0, 0, particle[2]])
            quat = r.as_quat()
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            pose_array.poses.append(pose)

        self.particles_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


        # super().__init__("particle_filter")

        # self.declare_parameter('particle_filter_frame', "particle_filter_frame")
        # self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        # self.declare_parameter('odom_topic', "/odom")
        # self.declare_parameter('scan_topic', "/scan")
        # self.declare_parameter("num_particles", "default") #check that this value is being imported correctly

        # scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        # odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        # self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)
        # self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        # self.pose_sub = self.create_subscription(PointStamped, "/clicked_point", self.pose_callback, 10)

        # #self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        # self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)
        # self.tf_pub = self.create_publisher(TransformStamped, "/tf", 1)

        # # Initialize the models
        # self.motion_model = MotionModel(self)
        # self.sensor_model = SensorModel(self)

        # self.get_logger().info("=============+READY+=============")

        # self.particles = None
        # self.initialized = False

        # self.particle_lock = threading.Lock()
    
    # def quaternion_to_yaw(self, quaternion):
    #     x = quaternion.x
    #     y = quaternion.y
    #     z = quaternion.z
    #     w = quaternion.w

    #     scipy_quaternion = (x, y, z, w)
    #     r = Rotation.from_quat(scipy_quaternion)
    #     euler_angles = r.as_euler('xyz')
    #     return euler_angles[2]