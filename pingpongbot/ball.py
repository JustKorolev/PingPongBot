import rclpy
import numpy as np
import random

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Pose, Vector3, Quaternion, Point
from std_msgs.msg               import ColorRGBA, Bool
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
        self.ball_pos_pub = self.create_publisher(Point, "/ball_pos", 1)
        self.ball_vel_pub = self.create_publisher(Point, "/ball_vel", 1)

        # Subscriptions
        self.create_subscription(Pose, "/tip_pose", self.tip_pose_callback, 1)
        self.create_subscription(Vector3, "/tip_vel", self.tip_vel_callback, 1)
        self.create_subscription(Bool, "/start", self.start_callback, 1)

        # Subscription variables
        self.tip_pos = np.zeros(3)
        self.tip_vel = np.zeros(3)
        self.tip_R = Reye()

        # Ball properties
        #unsure about this aproach but essentially a small raduis for collision detection
        self.radius = 0.02
        self.visual_radius = 0.02
        self.collision_tol = 0.1
        self.hit_timeout = 0
        self.wait_time = 0.0
        self.gravity = np.array([0.0, 0.0, -1.0])

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

        # Basket properties
        self.basket_marker = Marker()
        self.basket_marker.header.frame_id = "world"
        self.basket_marker.header.stamp = self.get_clock().now().to_msg()
        self.basket_marker.ns = "basket"
        self.basket_marker.id = 2
        self.basket_marker.type = Marker.CYLINDER
        self.basket_marker.action = Marker.ADD
        self.basket_marker.pose.position = Point(x=20.0, y=-20.0, z=0.0) # Adjust z to raise cylinder
        self.basket_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.basket_marker.scale = Vector3(x=0.5, y=0.5, z=0.1)
        self.basket_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        self.markerarray.markers.append(self.basket_marker)

        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)
        self.has_started = False

        # Create a timer to keep calling update().
        self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    def shutdown(self):
        self.destroy_node()

    def now(self):
        return self.start + Duration(seconds=self.t)

    def update(self):

        if not self.has_started:
            return

        pos_msg = Point(x=self.p[0], y=self.p[1], z=self.p[2])
        vel_msg = Point(x=self.v[0], y=self.v[1], z=self.v[2])
        self.ball_pos_pub.publish(pos_msg)
        self.ball_vel_pub.publish(vel_msg)

        # Update time
        self.t += self.dt

        # If we're waiting before respawning, just count down the wait.
        if self.wait_time > 0.0:
            self.wait_time -= self.dt
            if self.wait_time <= 0.0:
                # Time to respawn the ball
                self.spawn_ball()

        else:
            self.v += self.dt * self.a
            self.p += self.dt * self.v

            # Check if ball hit the ground
            if self.p[2] < self.radius:
                self.p[2] = self.radius
                self.v = np.zeros(3)
                self.wait_time = 1.0
                print(f"Final ball position: {self.p}")

        # Check for collision with paddle
        if self.check_hit() and self.hit_timeout <= 0:
            n = self.tip_R[:, 1]
            print(n)
            v_rel = self.v - self.tip_vel
            print(f"V rel: {v_rel}")
            print(f"Tip pos: {self.tip_pos}")
            self.v = self.v - (v_rel @ n) * n
            print(f"Tip velocity: {self.tip_vel}")
            print(f"ball velocity: {self.v}")
            self.a = self.gravity
            self.hit_timeout = 0.1

        self.hit_timeout -= self.dt
        self.marker.header.stamp  = self.now().to_msg()
        self.marker.pose.position.x = self.p[0]
        self.marker.pose.position.y = self.p[1]
        self.marker.pose.position.z = self.p[2]
        self.basket_marker.header.stamp = self.now().to_msg()  # Update basket timestamp
        self.marker_pub.publish(self.markerarray)


    def spawn_ball(self):
        # Respawn the ball at a random position and reset velocity
        self.p = np.array([0.0, 0.0, 0.1]) #self.generate_random_position()
        self.v = np.array([0.0, 0.0, 0.0])
        self.a = np.zeros(3)
        self.hit_timeout = 0

    def tip_pose_callback(self, pose):
        pos_array = np.array([pose.position.x, pose.position.y, pose.position.z])
        R = R_from_Quaternion(pose.orientation)
        self.tip_pos = pos_array
        self.tip_R = R

    def tip_vel_callback(self, vel):
        vel_array = np.array([vel.x, vel.y, vel.z])
        self.tip_vel = vel_array

    def start_callback(self, has_started):
        self.has_started = has_started

    def check_hit(self):
        abs_pos_diff = abs(self.p - self.tip_pos)
        tolerance_arr = np.ones(3) * (self.radius + self.collision_tol)
        result = np.less(abs_pos_diff, tolerance_arr)
        return np.equal(np.all(result), True)

    def generate_random_position(self):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        # limit height to 1.2 meters
        z = 1 #random.uniform(2, 4)
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
