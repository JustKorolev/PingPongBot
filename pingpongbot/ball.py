
"""balldemo.py

   Add deacription

   Node:      /balldemo
   Publish:   /visualization_marker_array   visualization_msgs.msg.MarkerArray

"""

import rclpy
import numpy as np
import random

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Pose, Vector3, Quaternion, Point
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

from pingpongbot.utils.TransformHelpers import *


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
        self.ball_vel_pub = self.create_publisher(Point, "/ball_vel", 10)

        # Subscriptions
        self.create_subscription(Pose, "/tip_pose", self.tip_pose_callback, 100)
        self.create_subscription(Vector3, "/tip_vel", self.tip_vel_callback, 100)

        # Subscription variables
        self.tip_pos = np.zeros(3)
        self.tip_vel = np.zeros(3)
        self.tip_R = Reye()

        # Ball properties
        #unsure about this aproach but essentially a small raduis for collision detection
        self.radius = 0.005
        self.visual_radius = 0.02
        self.collision_tol = 0
        self.hit_timeout = 0
        self.wait_time = 0.0
        self.gravity = np.array([0.0, 0.0, -9.81])

        # Spawn the ball initially
        self.spawn_ball()

        diam = 2 * self.visual_radius
        self.marker = Marker()
        self.marker.header.frame_id  = "world"
        self.marker.header.stamp     = self.get_clock().now().to_msg()
        self.marker.action           = Marker.ADD
        self.marker.ns               = "point"
        self.marker.id               = 1
        self.marker.type             = Marker.SPHERE
        self.marker.pose.orientation = Quaternion()
        self.marker.pose.position    = Point(x=self.p[0], y=self.p[1], z=self.p[2])
        self.marker.scale            = Vector3(x = diam, y = diam, z = diam)
        self.marker.color            = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)

        # Create the marker array message.
        self.markerarray = MarkerArray(markers = [self.marker])

        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create a timer to keep calling update().
        self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    def shutdown(self):
        self.destroy_node()

    def now(self):
        return self.start + Duration(seconds=self.t)

    def update(self):
        # Update time
        self.t += self.dt

        # If we're waiting before respawning, just count down the wait.
        if self.wait_time > 0.0:
            self.wait_time -= self.dt
            if self.wait_time <= 0.0:
                # Time to respawn the ball
                self.spawn_ball()

        else:
            self.a = self.gravity
            self.v += self.dt * self.a
            self.p += self.dt * self.v

            # Check if ball hit the ground
            if self.p[2] < self.radius:
                self.p[2] = self.radius
                self.v = np.zeros(3)
                self.wait_time = 1.0

        # Check for collision with paddle
        if self.check_hit() and self.hit_timeout <= 0:
            n = self.tip_R[:, 1]
            print(self.tip_vel)
            print(n)
            v_rel = self.v - self.tip_vel
            self.v = self.v - 2 * (v_rel @ n) * n
            print(self.v)
            self.hit_timeout = 0.5

        self.hit_timeout -= self.dt
        self.marker.header.stamp  = self.now().to_msg()
        self.marker.pose.position.x = self.p[0]
        self.marker.pose.position.y = self.p[1]
        self.marker.pose.position.z = self.p[2]
        self.marker_pub.publish(self.markerarray)
        pos_msg = Point(x=self.p[0], y=self.p[1], z=self.p[2])
        vel_msg = Point(x=self.v[0], y=self.v[1], z=self.v[2])
        self.ball_pos_pub.publish(pos_msg)
        self.ball_vel_pub.publish(vel_msg)


    def spawn_ball(self):
        # Respawn the ball at a random position and reset velocity
        self.p = self.generate_random_position()
        self.v = np.array([0.0, 0.0, 0.0])
        self.hit_timeout = 0

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
        y = random.uniform(0.4, 1.0)
        # limit height to 1.2 meters
        z = random.uniform(0.4, 1.2)
        return np.array([x, y, z])

#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the demo node
    rclpy.init(args=args)
    node = DemoNode('ball', 1000)

    # Run until interrupted.
    rclpy.spin(node)

    # Shutdown
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
