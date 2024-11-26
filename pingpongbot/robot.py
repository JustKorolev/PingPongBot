import rclpy
import numpy as np
import random

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from hw6code.KinematicChain     import KinematicChain
 # Import the format for the condition number message
from geometry_msgs.msg import Pose, Vector3, Point

#
# Trajectory Class
#
class Trajectory():
    # Initialization
    def __init__(self, node):

        self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())

        # Our home joint angles
        self.home_q = np.array([-0.1, -2.2, 2.4, -1.0, 1.6, 1.6])
        self.q_centers = [-pi/2, -pi/2, 0, -pi/2, 0, 0]

        # Initial position
        self.q0 = self.home_q
        self.home_p, self.home_R, _, _  = self.chain.fkin(self.q0)
        self.p0 = self.home_p
        self.R0 = self.home_R

        # Swing back variables
        self.swing_back_time = 2
        self.R_swing_back = R_from_RPY(0, 0, 0)
        self.swing_rot_axis_array, self.swing_rot_angle = axisangle_from_R(self.R_swing_back - self.home_R)
        self.swing_rot_axis = nxyz(self.swing_rot_axis_array[0], self.swing_rot_axis_array[1], self.swing_rot_axis_array[2])

        # Swing variables
        self.hit_time = 1.5
        self.return_time = 1.5
        self.hit_pos = np.zeros(3)

        self.qd = self.q0
        self.pd = self.p0
        self.Rd = self.R0

        # Tuning constants
        self.lam = 20
        self.lam_second = 5
        self.gamma = 0.1
        self.gamma_array = [self.gamma ** 2] * len(self.jointnames())

        # Publishing
        self.tip_pose_pub = node.create_publisher(Pose, "/tip_pose", 10)
        self.tip_vel_pub = node.create_publisher(Vector3, "/tip_vel", 10)

        # Subscribing
        node.create_subscription(Point, "/ball_pos", self.ball_pos_callback, 10)

        # Subscription variables
        self.ball_pos = np.zeros(3)


    def jointnames(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def evaluate(self, t, dt):
        # TODO: change to whatever it should be dynamically
        # Desired hit position
        # Desired swing back position
        swing_back_pd = self.ball_pos - np.array([0, 1, 0.15])
        desired_hit_velocity = np.array([random.random() for _ in range(3)])
        cycle_time = self.swing_back_time + self.hit_time + self.return_time

        # Swing back sequence
        if t < self.swing_back_time:
            pd, vd = spline(t, self.swing_back_time, self.home_p, swing_back_pd,
                            np.zeros(3), np.zeros(3))

        # Hit sequence
        elif t < self.swing_back_time + self.hit_time:
            pd, vd = spline(t - self.swing_back_time, self.hit_time,
                            swing_back_pd, self.ball_pos, np.zeros(3), desired_hit_velocity)
            self.hit_pos = self.ball_pos

        # Return home sequence
        elif t < self.swing_back_time + self.hit_time + self.return_time:
            pd, vd = spline(t - (self.swing_back_time + self.hit_time), self.return_time,
                            self.hit_pos, self.home_p, desired_hit_velocity, np.zeros(3))



        # TODO: TEMPORARY UNCHANGING ROTATION -- CHANGE
        Rd = self.R0
        wd = np.zeros(3)


        qdlast = self.qd
        pdlast = self.pd
        Rdlast = self.Rd

        (pr, Rr, Jv, Jw) = self.chain.fkin(qdlast)
        error_r = eR(Rdlast, Rr)
        error_p = ep(pdlast, pr)

        adjusted_vd = vd + (self.lam * error_p)
        adjusted_wd = wd + (self.lam * error_r)

        # TREATING WD AS A SECONDARY TASK
        jac_p = Jv
        jac_s = Jw
        jac_p_pinv = np.linalg.pinv(jac_p)
        jac_s_pinv = np.linalg.pinv(jac_s)

        N = jac_p.shape[1]
        qddot = jac_p_pinv @ adjusted_vd
        qsdot = jac_s_pinv @ adjusted_wd
        qddot = jac_p_pinv @ vd +\
            (np.eye(N) - jac_p_pinv @ jac_p) @ qsdot


        # WORKED POORLY -----------------------------------
        # combined_v_vec = np.concatenate((adjusted_vd, adjusted_wd))

        # jac = np.vstack((Jv, Jw))
        # jac_p_pinv = np.linalg.pinv(jac)
        # jac_winv = np.linalg.pinv(jac.T @ jac + np.diag(self.gamma_array)) @ jac.T

        # N = jac.shape[1]
        # qsdot = self.lam_second * (self.q_centers - qdlast)
        # qddot = jac_p_pinv @ combined_v_vec +\
        #             (np.eye(N) - jac_winv @ jac) @ qsdot

        qd = qdlast + dt * qddot

        self.qd = qd
        self.pd = pd
        self.Rd = Rd

        # Publishing
        self.tip_pose_msg = self.create_pose(self.pd, self.Rd)
        self.tip_vel_msg = self.create_vel_vec(adjusted_vd)
        self.tip_pose_pub.publish(self.tip_pose_msg)
        self.tip_vel_pub.publish(self.tip_vel_msg)

        return (qd, qddot, pd, vd, Rd, wd)


    # Takes a numpy array position and R matrix to produce a ROS pose msg
    def create_pose(self, position, orientation):
        pose = Pose()
        pose.position = Point_from_p(position)
        pose.orientation = Quaternion_from_R(orientation)
        return pose


    # Takes a numpy array velocity and returns a ROS vec3 message
    def create_vel_vec(self, velocity):
        vx, vy, vz = list(velocity)
        vec3 = Vector3(x=vx, y=vy, z=vz)
        return vec3


    def ball_pos_callback(self, pos):
        self.ball_pos = p_from_Point(pos)



def main(args=None):
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
