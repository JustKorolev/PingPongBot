"""balldemo.py

   Simulate a non-physical ball and publish as a visualization marker
   array to RVIZ.

   Node:      /balldemo
   Publish:   /visualization_marker_array   visualization_msgs.msg.MarkerArray

"""

import rclpy
import numpy as np
import random

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Pose, Vector3, Quaternion
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

from pingpongbot.utils.TransformHelpers     import *


#
#   Demo Node Class
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name, rate):

        # Initialize the node, naming it as specified
        super().__init__(name)

        # Prepare the publisher (latching for new subscribers).
        quality = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.marker_pub = self.create_publisher(MarkerArray,
                                        '/visualization_marker_array', quality)
        self.ball_pos_pub = self.create_publisher(Point, "/ball_pos", 10)

        # Subscribing
        self.create_subscription(Pose, "/tip_pose", self.tip_pose_callback, 100)
        self.create_subscription(Vector3, "/tip_vel", self.tip_vel_callback, 100)

        # Subscription variables
        self.tip_pos = np.zeros(3)
        self.tip_vel = np.zeros(3)
        self.tip_R = Reye()

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.02
        self.collision_tol = 0
        self.hit_timeout = 0

        self.p = self.generate_random_position()
        self.v = np.array([0.0, 0.0, 0.0])
        self.a = np.array([0.0, 0.0, 0.0])

        # Create the sphere marker.
        diam        = 2 * self.radius
        self.marker = Marker()
        self.marker.header.frame_id  = "world"
        self.marker.header.stamp     = self.get_clock().now().to_msg()
        self.marker.action           = Marker.ADD
        self.marker.ns               = "point"
        self.marker.id               = 1
        self.marker.type             = Marker.SPHERE
        self.marker.pose.orientation = Quaternion()
        self.marker.pose.position    = Point_from_p(self.p)
        self.marker.scale            = Vector3(x = diam, y = diam, z = diam)
        self.marker.color            = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        # a = 0.8 is slightly transparent!

        # Create the marker array message.
        self.markerarray = MarkerArray(markers = [self.marker])

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create a timer to keep calling update().
        self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the node, including cleaning up the timer.
        self.destroy_node()

    # Return the current time (in ROS format).
    def now(self):
        return self.start + Duration(seconds=self.t)

    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Integrate the velocity, then the position.
        self.v += self.dt * self.a
        self.p += self.dt * self.v

        # Check for a bounce - note the change in x velocity is non-physical.
        if self.p[2] < self.radius:
            self.p[2] = self.radius + (self.radius - self.p[2])
            self.v[2] *= -1.0
            self.v[0] *= -1.0   # Change x just for the fun of it!

        # Check for collision with paddle
        if self.check_hit() and self.hit_timeout <= 0:
            n = self.tip_R[:, 1]
            print(self.tip_vel)
            v_rel = self.v - self.tip_vel
            self.v = self.v - 2 * (v_rel @ n) * n
            print(self.v)
            self.hit_timeout = 2

        # Subtract from hit timeout
        self.hit_timeout -= self.dt

        # Update the ID number to create a new ball and leave the
        # previous balls where they are.
        #####################
        # self.marker.id += 1
        #####################

        # Update the message and publish.
        self.marker.header.stamp  = self.now().to_msg()
        self.marker.pose.position = Point_from_p(self.p)
        self.marker_pub.publish(self.markerarray)
        self.ball_pos_pub.publish(self.marker.pose.position)


    def tip_pose_callback(self, pose):
        pos_array = np.array([pose.position.x, pose.position.y, pose.position.z])
        R = R_from_Quaternion(pose.orientation)
        self.tip_pos = pos_array
        self.tip_R = R


    def tip_vel_callback(self, vel):
        vel_array = np.array([vel.x, vel.y, vel.z])
        self.tip_vel = vel_array


    def check_hit(self):
        abs_pos_diff = abs(self.p - self.tip_pos)
        tolerance_arr = np.ones(3) * (self.radius + self.collision_tol)
        result = np.less(abs_pos_diff, tolerance_arr)
        return np.equal(np.all(result), True)


    def generate_random_position(self):
        x = random.uniform(-1, 1)
        y = random.uniform(0.4, 1)
        z = random.uniform(0.4, 1)
        return np.array([x, y, z])

#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the demo node (100Hz).
    rclpy.init(args=args)
    node = DemoNode('ball', 1000)

    # Run until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
