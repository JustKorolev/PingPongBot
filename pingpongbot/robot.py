import rclpy
import numpy as np
import random

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from pingpongbot.utils.KinematicChain     import KinematicChain
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
        self.q_centers = np.array([-pi/2, -pi/2, 0, -pi/2, 0, 0])

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
        self.swing_back_q = None

        # Swing variables
        self.hit_time = 1.5
        self.return_time = 1.5
        self.hit_pos = np.zeros(3)

        self.qd = self.q0
        self.pd = self.p0
        self.Rd = self.R0

        # Tuning constants
        self.lam = 25
        self.lam_second = 15
        self.gamma = 0.1
        self.gamma_array = [self.gamma ** 2] * len(self.jointnames())

        # Publishing
        self.tip_pose_pub = node.create_publisher(Pose, "/tip_pose", 10)
        self.tip_vel_pub = node.create_publisher(Vector3, "/tip_vel", 10)

        # Subscribing
        node.create_subscription(Point, "/ball_pos", self.ball_pos_callback, 10)

        # Subscription variables #TODO FIX THIS TO DEFAULT
        self.ball_pos = np.array([0.8, 0.8, 0.8])


    def jointnames(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def adjust_jacobian(self, Jv, Jw):
        J_combined = np.vstack((Jv, Jw))
        # Remove the last row
        J_adjusted = J_combined[:-1, :]
        return J_adjusted

    def evaluate(self, t, dt):
        pd = self.pd
        cycle_time = self.swing_back_time + self.hit_time + self.return_time
        swing_back_pd = self.ball_pos - np.array([0, 1, 0.15])
        desired_hit_velocity = np.array([0.5, 0.5, 0.5])

        # Swing back sequence
        # if t < self.swing_back_time:
        #     pd, vd = spline(t, self.swing_back_time, self.home_p, swing_back_pd,
        #                     np.zeros(3), np.zeros(3))
        #     self.swing_back_q = self.qd

        # Hit sequence
        if t < self.hit_time:
            qf = self.newton_raphson(self.ball_pos, self.home_q) # TODO: want to do this only once
            qd, qddot = spline(t, self.hit_time, self.home_q, qf, np.zeros(6), np.ones(6))
            self.hit_pos = self.ball_pos

            # FOR TESTING
            pd, Rd, Jv, _ = self.chain.fkin(qd)
            vd = np.zeros(3)

        # Return home sequence
        # elif t < self.hit_time + self.return_time:
        #     pd, vd = spline(t - (self.hit_time), self.return_time,
        #                     self.hit_pos, self.home_p, desired_hit_velocity, np.zeros(3))



        # TODO: TEMPORARY UNCHANGING ROTATION -- CHANGE
        wd = np.zeros(3)

        # Kinematics
        qdlast = self.qd
        pdlast = self.pd
        Rdlast = self.Rd
        pr, Rr, Jv, Jw = self.chain.fkin(qdlast)


        #print("Desired Rotation Matrix:\n", Rd)

        # Position and rotation errors
        error_p = ep(pdlast, pr)
        error_r = eR(Rdlast, Rr)

        # Adjusted velocities
        adjusted_vd = vd + (self.lam * error_p)
        adjusted_wd = (wd + (self.lam * error_r))[:2]
        combined_vwd = np.concatenate([adjusted_vd, adjusted_wd])

        # Jacobian adjustments
        J_adjusted = self.adjust_jacobian(Jv, Jw)
        J_pinv = np.linalg.pinv(J_adjusted)

        # Primary task
        qddot_main = J_pinv @ combined_vwd

        # Secondary task
        q_centers_error = self.q_centers - qdlast
        qddot_secondary = self.lam_second * q_centers_error
        N = J_adjusted.shape[1]

        if not (t < self.hit_time):
            qddot = qddot_main + (np.eye(N) - J_pinv @ J_adjusted) @ qddot_secondary
        qd = qdlast + dt * qddot

        # Update state
        self.qd = qd
        self.pd = pd
        self.Rd = Rd

        # Publishing
        self.tip_pose_msg = self.create_pose(self.pd, self.Rd)
        self.tip_vel_msg = self.create_vel_vec(adjusted_vd)
        self.tip_pose_pub.publish(self.tip_pose_msg)
        self.tip_vel_pub.publish(self.tip_vel_msg)

        return (qd, qddot, pd, vd, Rd, wd)

    # Newton Raphson
    def newton_raphson(self, pgoal, q0):

        # Collect the distance to goal and change in q every step!
        pdistance = []
        qstepsize = []

        # Number of steps to try.
        N = 40

        # Setting initial q
        q = q0

        # IMPLEMENT THE NEWTON-RAPHSON ALGORITHM!
        for _ in range(N):
            (pr, _, Jv, Jw) = self.chain.fkin(q)
            jac = Jv #np.vstack((Jv, Jw))
            # TODO INSERT MODIFIED JACOBIAN HERE, ADD DESIRED ORIENTATION
            q_new = q + np.linalg.pinv(jac) @ (pgoal - pr)
            qstepsize.append(np.linalg.norm(q_new - q))
            pdistance_curr = np.linalg.norm(pgoal - pr)
            pdistance.append(pdistance_curr)
            q = q_new

        # Unwrap
        for i in range(len(q)):
            q[i] = fmod(q[i], 2*pi)

        return q

    # Takes a numpy array position and R matrix to produce a ROS pose msg
    def create_pose(self, position, orientation):
        pose = Pose()
        pose.position = Point_from_p(position)
        pose.orientation = Quaternion_from_R(orientation)
        return pose

    def create_vel_vec(self, velocity):
        vx, vy, vz = velocity
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
