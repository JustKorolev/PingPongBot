import rclpy
import numpy as np
import random

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from pingpongbot.utils.GeneratorNode      import GeneratorNode
from pingpongbot.utils.TransformHelpers   import *
from pingpongbot.utils.TrajectoryUtils    import *

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
        self.home_p, self.home_R, _, _  = self.chain.fkin(self.home_q)
        self.p0 = self.home_p
        self.R0 = self.home_R

        # Swing back variables
        self.swing_back_time = 2
        self.R_swing_back = R_from_RPY(0, 0, 0)
        self.swing_rot_axis_array, self.swing_rot_angle = axisangle_from_R(self.R_swing_back - self.home_R)
        self.swing_rot_axis = nxyz(self.swing_rot_axis_array[0], self.swing_rot_axis_array[1], self.swing_rot_axis_array[2])
        self.swing_back_q = None

        # Swing variables
        self.hit_time = float("inf") # TODO: SHOULD BE SCALED TO PATH DISTANCE
        self.return_time = float("inf")
        self.hit_pos = np.zeros(3)
        self.hit_rotation = Reye()
        self.hit_q = np.zeros(6)

        self.qd = self.q0
        self.pd = self.p0
        self.Rd = self.R0

        # Tuning constants
        self.lam = 20
        self.lam_second = 15
        self.gamma = 0.08
        self.gamma_array = [self.gamma ** 2] * len(self.jointnames())

        # Publishing
        self.tip_pose_pub = node.create_publisher(Pose, "/tip_pose", 100)
        self.tip_vel_pub = node.create_publisher(Vector3, "/tip_vel", 100)

        # Subscribing
        node.create_subscription(Point, "/ball_pos", self.ball_pos_callback, 10)

        # Subscription variables
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
        desired_hit_velocity = np.array([2, 2, 2])

        # Swing back sequence
        # if t < self.swing_back_time:
        #     pd, vd = spline(t, self.swing_back_time, self.home_p, swing_back_pd,
        #                     np.zeros(3), np.zeros(3))
        #     self.swing_back_q = self.qd

        # Hit sequence
        if t < self.hit_time:
            # TODO: want to do this only once
            R_rel = self.home_R.T @ self.hit_rotation
            theta = fmod(np.arccos((np.trace(R_rel) - 1) / 2), 2*pi)
            rot_axis = np.array([
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1]
                ]) / (2 * np.sin(theta))
            _, _, Jvf, _ = self.chain.fkin(self.hit_q)
            qdotf = np.linalg.pinv(Jvf) @ desired_hit_velocity
            if t < dt: # TODO: TENTATIVE FIX
                # ______________________________________
                Rf = R_from_RPY(-pi/4, 0, 5*pi/4) # TODO: TEMPORARY FOR TESTING, FIGURE OUT RELATIVE TO WORLD
                self.hit_rotation = Rf
                self.hit_pos = self.ball_pos # TODO: REDUNDANT

                self.hit_q = self.newton_raphson(self.ball_pos, self.home_q)

                self.hit_time = self.calculate_sequence_time(self.qd, self.hit_q, np.zeros(6), qdotf)
                #____________________________________________________

            # ROTATION CALCULATION
            sr, srdot = goto(t, self.hit_time, 0, 1)
            Rot = rodrigues_formula(rot_axis, theta * sr)
            Rd = self.home_R @ Rot
            wd = self.home_R @ rot_axis * (theta * srdot) # TODO: MAY BE ILL DEFINED SINCE THERE ARE DISCREPANCIES

            # Pure position manipulation
            qd_hit, qddot_hit = spline(t, self.hit_time, self.home_q, self.hit_q, np.zeros(6), qdotf)

            pd, _, Jv, _ = self.chain.fkin(qd_hit)
            vd = Jv @ qddot_hit

        # Return home sequence
        elif t < self.hit_time + self.return_time:
            if t < self.hit_time + dt: # TODO: TENTATIVE FIX
                self.return_time = 1 # TODO: MAKE DYNAMIC LIKE SELF.HIT_TIME
            #TODO: DO ONLY ONCE
            # _____________________

            R_rel = self.hit_rotation.T @ self.home_R
            theta = fmod(np.arccos((np.trace(R_rel) - 1) / 2), 2*pi)
            rot_axis = np.array([
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1]
                ]) / (2 * np.sin(theta))
            #________________________


            # POSITION CALCULATION
            pd, vd = spline(t - self.hit_time, self.return_time,
                            self.hit_pos, self.home_p, desired_hit_velocity, np.zeros(3))

            # print(pd)
            # print(vd)

            # ROTATION CALCULATION

            sr, srdot = goto(t - self.hit_time, self.return_time, 0, 1)
            Rot = rodrigues_formula(rot_axis, theta * sr)
            Rd = self.hit_rotation @ Rot
            wd = self.hit_rotation @ (rot_axis * theta * srdot)
        else:
            pd = self.pd
            vd = np.zeros(3)
            Rd = self.Rd
            wd = np.zeros(3)



        # TODO: TEMPORARY UNCHANGING ROTATION -- CHANGE
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
        # adjusted_wd = (wd + (self.lam * error_r))[:2]
        adjusted_wd = (wd + (self.lam * error_r))
        combined_vwd = np.concatenate([adjusted_vd, adjusted_wd])

        # Jacobian adjustments
        # J_adjusted = self.adjust_jacobian(Jv, Jw)
        J_adjusted = np.vstack([Jv, Jw])
        J_p = J_adjusted[:3, :]
        J_s = J_adjusted[3:, :]
        J_pinv_p = np.linalg.pinv(J_p)
        J_pinv_s = np.linalg.pinv(J_s)
        J_pinv = np.linalg.pinv(J_adjusted)

        # Primary task
        qddot_main = J_pinv_p @ adjusted_vd

        # Secondary task
        qddot_secondary = J_pinv_s @ adjusted_wd
        N = J_adjusted.shape[1]

        # BASIC QDDOT CALCULATION
        # TODO: CONSIDER USING TARGETED-REMOVAL/BLENDING
        jac_winv = np.linalg.pinv(J_adjusted.T @ J_adjusted +\
                                np.diag(self.gamma_array)) @ J_adjusted.T
        qddot = jac_winv @ combined_vwd

        # MORE SOPHISTICATED QDDOT CALCULATIONS
        # if not (t < self.hit_time):
        #     qddot = qddot_main + (np.eye(N) - J_pinv_p @ J_p) @ qddot_secondary
        # else:
        #     qddot = J_pinv @ combined_vwd
        #     # qddot = qddot_hit + (np.eye(N) - J_pinv_p @ J_p) @ qddot_secondary

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
        N = 100

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


    # TODO: MAKE THIS MORE SOPHISTICATED
    def calculate_sequence_time(self, q0, qf, qddot0, qddotf):
        # TODO: THIS IS VERY JANK
        avg_qddot = np.linalg.norm(qddotf - qddot0) / 8
        print(avg_qddot)
        return np.linalg.norm(qf - q0) / avg_qddot


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
    generator = GeneratorNode('generator', 200, Trajectory)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
