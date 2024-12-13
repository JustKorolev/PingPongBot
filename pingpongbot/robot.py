import time
import rclpy
import numpy as np
import random
from scipy.optimize import minimize

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from pingpongbot.utils.GeneratorNode      import GeneratorNode
from pingpongbot.utils.TransformHelpers   import *
from pingpongbot.utils.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from pingpongbot.utils.KinematicChain     import KinematicChain
 # Import the format for the condition number message
from geometry_msgs.msg import Pose, Vector3, Point
from std_msgs.msg import Bool

#
# Trajectory Class
#
class Trajectory():
    # Initialization
    def __init__(self, node):

        # Publishing
        self.tip_pose_pub = node.create_publisher(Pose, "/tip_pose", 1)
        self.tip_vel_pub = node.create_publisher(Vector3, "/tip_vel", 1)
        self.start_pub = node.create_publisher(Bool, "/start", 1)
        self.start_pub.publish(Bool())

        # Subscribing
        node.create_subscription(Point, "/ball_pos", self.ball_pos_callback, 1)
        node.create_subscription(Point, "/ball_vel", self.ball_vel_callback, 1)

        # Subscription variables
        self.ball_pos = np.ones(3)
        self.ball_vel = np.zeros(3)

        # Forward kinematics chain
        self.tip_chain = KinematicChain(node, 'world', 'tip', self.jointnames())
        self.elbow_chain = KinematicChain(node, 'world', 'forearm_link', self.jointnames()[:-3])
        self.shoulder_lift_chain = KinematicChain(node, 'world', 'upper_arm_link', self.jointnames()[:-4])

        #BALL AND BOWL
        self.home_q = np.array([0, -pi/2, pi/2, pi/2, 0, 0])
        # self.home_q = np.array([0, 0, -5*pi/6, -pi, 0, 0])
        # self.home_q = np.array([-0.1, -2.2, 2.4, -1.0, 1.6, 1.6])
        self.q_centers = np.array([-pi/2, -pi/2, 0, -pi/2, 0, 0])

        # Initial position
        self.q0 = self.home_q
        self.home_p, self.home_R, _, _  = self.tip_chain.fkin(self.home_q)
        self.p0 = self.home_p
        self.R0 = self.home_R
        self.qdot0 = np.zeros(6)
        self.paddle_hit_vel = np.zeros(3)
        self.paddle_hit_normal = np.zeros(3)

        # Swing variables
        self.hit_time = float("inf")
        self.return_time = float("inf")
        self.hit_pos = np.zeros(3)
        self.q_hit = np.zeros(6)
        self.qdot_hit = np.zeros(6)
        self.time_offset = 0

        # Kinematics variables
        self.qd = self.q0
        self.vd = np.zeros(3)
        self.pd = self.p0
        self.Rd = self.R0
        self.nd = self.get_paddle_normal(self.R0)
        self.qddot = np.zeros(3)

        # Actual kinematics variable
        self.actual_q_hit = np.zeros(6)

        # Tuning constants
        self.lam = 20
        self.gamma = 0.3
        self.repulsion_const = 0
        self.gamma_array = [self.gamma ** 2] * len(self.jointnames())
        self.max_joint_vels = np.array([4, 4, 4, 4, 4, 4])
        self.weight_matrix = np.diag(self.max_joint_vels ** 2)

        # Robot parameters
        self.max_reach_rad = 1.3
        self.robot_base_position = np.array([0, 0, 0.1])



    def jointnames(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


    def adjusted_jacobian(self, Jv, Jw, nr):
        n_cross = crossmat(nr)
        Jn = n_cross @ Jw
        J_adjusted = np.vstack([Jv, Jn])
        return J_adjusted


    def evaluate(self, t, dt):

        # TODO: TESTING
        self.hit_pos = np.array([0.5, 0, -0.5]) # Robot reach radius is 1.3m
        self.ball_hit_velocity = np.zeros(3)
        ball_target_pos = np.array([20, -20, 0])
        g = np.array([0.0, 0.0, -1.0])  # Adjust magnitude as needed, e.g., -9.81 for real gravity

        # Hit sequence
        if t - self.time_offset < self.hit_time:

            # PRE CALCULATIONS FOR TRAJECTORY
            if t - self.time_offset < dt:

                if self.hit_pos[2] < 0:
                    self.repulsion_const = 15 # TODO: FIX
                else:
                    self.repulsion_const = 15

                if np.linalg.norm(self.hit_pos) > self.max_reach_rad:
                    print("WARNING: OBJECT OUTSIDE OF WORKSPACE. ROBOT WILL NOT REACH.")

                # Calculate the required paddle velocity and normal to hit ball in basket
                self.paddle_hit_vel = self.calculate_min_paddle_vel(self.hit_pos, ball_target_pos, g)
                self.paddle_hit_normal = self.paddle_hit_vel / np.linalg.norm(self.paddle_hit_vel)


                print(f"Computed Paddle Hit Velocity: {self.paddle_hit_vel}")

                # Use newton raphson to converge to the final joint angles under task constraints
                self.q_hit = self.newton_raphson(self.hit_pos, self.paddle_hit_normal, self.home_q)
                _, _, Jvf, _ = self.tip_chain.fkin(self.q_hit)
                self.qdot_hit = np.linalg.pinv(Jvf) @ self.paddle_hit_vel

                # Calculate the trajectory time
                self.hit_time = self.calculate_sequence_time(self.q0, self.q_hit)
                self.return_time = self.hit_time

            if t - self.time_offset > self.hit_time - dt:
                # Publishing at moment before impact
                self.tip_pose_msg = self.create_pose(self.pd, self.Rd)
                self.tip_vel_msg = self.create_vel_vec(self.vd)
                self.tip_pose_pub.publish(self.tip_pose_msg)
                self.tip_vel_pub.publish(self.tip_vel_msg)
                self.actual_q_hit = self.qd

                # TODO: TROUBLESHOOTING VALUES AT IMPACT
                print(f"ACTUAL FINAL PADDLE VEL: {self.vd}")
                print(f"ACTUAL PADDLE NORMAL: {self.nd}")
                print(f"ACTUAL HIT POSITION: {self.pd}")
                print(f"ACTUAL JOINT POSITION: {self.qd}")
                print(f"DESIRED PADDLE VELOCITY: {self.paddle_hit_vel}")
                print(f"DESIRED PADDLE NORMAL: {self.paddle_hit_normal}")
                print(f"DESIRED HIT POSITION: {self.hit_pos}")
                print(f"DESIRED JOINT POSITION: {self.q_hit}")

            # qd calculation using trajectory in joint space
            qd_hit, qddot_hit = spline(t - self.time_offset, self.hit_time, self.q0, self.q_hit, self.qdot0, self.qdot_hit)

            # Calculating robot values using forward kinematics
            pd, Rd, Jv, Jw = self.tip_chain.fkin(qd_hit)
            vd = Jv @ qddot_hit
            wd = Jw @ qddot_hit
            nd = self.get_paddle_normal(Rd)

        # Return home sequence
        # TODO: CHECK WHY ROBOT SOMETIMES DOESNT RETURN TO CORRECT HOME ANGLES
        elif t - self.time_offset - self.hit_time < self.return_time:
            qd_return, qddot_return = spline(t - self.time_offset - self.hit_time, self.return_time, self.q_hit, self.home_q, self.qdot_hit, np.zeros(6))
            pd, Rd, Jv, Jw = self.tip_chain.fkin(qd_return)
            vd = Jv @ qddot_return
            wd = Jw @ qddot_return
            nd = self.get_paddle_normal(Rd)

        else:
            return
            # qd_settle = self.home_q
            # pd, Rd, Jv, Jw = self.tip_chain.fkin(qd_settle)
            # vd = Jv @ qd_settle
            # wd = Jw @ qd_settle
            # nd = self.get_paddle_normal(Rd)


        # Kinematics
        qdlast = self.qd
        pdlast = self.pd
        Rdlast = self.Rd
        ndlast = self.get_paddle_normal(Rdlast)
        pr, Rr, Jv, Jw = self.tip_chain.fkin(qdlast)
        nr = self.get_paddle_normal(Rr)

        # Position and normal errors
        error_p = ep(pdlast, pr)
        error_n = ep(ndlast, nr)

        # Adjusted velocities
        adjusted_vd = vd + ((self.lam * error_p) - (0.0 * error_n/dt))
        adjusted_nd = nd - ((self.lam * error_n) - (0.0 * error_n/dt))
        combined_vnd = np.concatenate([adjusted_vd, adjusted_nd])

        # Jacobian adjustments
        Jp = self.adjusted_jacobian(Jv, Jw, nd)

        # Primary task
        # qddot_main = J_pinv_p @ adjusted_vd

        # Secondary task
        qsdot = self.repulsion(qdlast)
        N = Jp.shape[0]

        # PRIMARY TASK: VELOCITY AND PADDLE ORIENTATION
        # Additionally, regularization and max joint elocity constraints.
        # TODO: NOT SURE EXACTLY HOW THE WEIGHTING WORKS
        Jp_pinv = np.linalg.pinv(Jp.T @ Jp +\
                                np.diag(self.gamma_array)) @ Jp.T

        # Jp_pinv = np.linalg.pinv(Jp.T @ self.weight_matrix @ Jp +\
        #                         np.diag(self.gamma_array)) @ Jp.T @ self.weight_matrix


        # PRIMARY TASK: VELOCITY AND PADDLE ORIENTATION
        # SECONDARY TASK: REPULSION FROM FLOOR

        qddot = Jp_pinv @ combined_vnd +\
                    (np.eye(N) - Jp_pinv @ Jp) @ qsdot

        # print((np.eye(N) - jac_winv @ Jp) @ qsdot)

        # QDDOT WITH JOINT WEIGHTING MATRIX
        # jac_weighted = self.weight_matrix @ jac_winv.T @\
        #     np.linalg.pinv(jac_winv @ self.weight_matrix @ jac_winv.T)
        # qddot = np.linalg.pinv(self.weight_matrix) @ jac_winv.T @ \
        #        np.linalg.pinv(jac_winv @ np.linalg.pinv(self.weight_matrix) @ jac_winv.T) @ combined_vwd




        qd = qdlast + dt * qddot

        # Update state
        self.qd = qd
        self.pd = pd
        self.Rd = Rd
        self.vd = vd
        self.nd = nd
        self.qddot = qddot

        return (qd, qddot, pd, vd, Rd, wd)


    def repulsion(self, q):
        # Compute the wrist and elbow points.
        (p_elbow, _, Jv_elbow, Jw_elbow) = self.elbow_chain.fkin(q[:3])  # 3 joints
        (p_shoulder, _, _, _) = self.shoulder_lift_chain.fkin(q[:2])  # 2 joints
        (p_tip, _, _, _) = self.tip_chain.fkin(q)

        # Calculate "distance" between elbow and ground
        elbow_distance_to_ground = p_elbow[2]
        tip_distance_to_ground = p_tip[2]
        tip_base_diff = p_tip - self.robot_base_position
        tip_distance_to_base = np.linalg.norm(tip_base_diff)
        tip_base_diff_directions = np.sign(tip_base_diff)

        # Repulsion based on exponential curve
        c = self.repulsion_const
        F_elbow = np.array([0, 0, np.e**(c*(-elbow_distance_to_ground + 0.5))])
        F_tip_ground = np.array([0, 0, np.e**(c*(-tip_distance_to_ground + 0.5))])
        F_tip_base = np.array([np.e**(0.6*c*(-tip_distance_to_base + 0.6))] * 3) * tip_base_diff_directions
        print(F_tip_base)
        F_total = F_elbow + F_tip_ground + F_tip_base

        # Map the repulsion force acting at parm to the equivalent force
        # and torque actiing at the wrist point.
        T_elbow = np.cross(p_shoulder - p_elbow , F_elbow)
        T_tip_ground = np.cross(p_tip - p_elbow , F_tip_ground)
        T_tip_base = np.cross(p_tip - self.robot_base_position, F_tip_base)
        T_total = T_elbow + T_tip_ground + T_tip_base

        J_stacked = np.vstack((Jv_elbow, Jw_elbow))
        force_torque_stacked = np.concatenate((F_total, T_total))
        # print(F_total)

        # Convert the force/torque to joint torques (J^T).
        tau = J_stacked.T @ force_torque_stacked
        # print(tau)

        # Return the 2 joint torques as part of the 6 full joints.
        return np.concatenate([tau, np.zeros(3)])


    def calculate_shortest_angle(self, q0, qf):

        angle_diff = qf - q0
        wrapped_delta = fmod(angle_diff + pi, 2*pi) - pi
        shortest_angle = q0 + wrapped_delta

        return shortest_angle


    # Newton Raphson
    def newton_raphson(self, pgoal, ngoal, q0, shortest_angles=True):
        # Number of steps to try.
        N = 100

        # Setting initial q
        q = q0

        for _ in range(N):
            (pr, R, Jv, Jw) = self.tip_chain.fkin(q)
            nr = self.get_paddle_normal(R)
            jac = self.adjusted_jacobian(Jv, Jw, nr)
            p_error = pgoal - pr
            n_error = ngoal - nr
            combined_error = np.concatenate([p_error, n_error])

            q = q + np.linalg.pinv(jac) @ (combined_error)

        # Unwrap q
        for i in range(len(q)):
            q[i] = fmod(q[i], 2*pi)

            if shortest_angles:
                q[i] = self.calculate_shortest_angle(self.q0[i], q[i])



        return q


    def calculate_sequence_time(self, q0, qf):
        max_sequence_time = 5
        q_diff_list = list(abs(qf - q0))
        max_q_diff = max(q_diff_list)
        sequence_time = 3*(max_q_diff) / self.max_joint_vels[q_diff_list.index(max_q_diff)]
        capped_sequence_time = min(max_sequence_time, sequence_time)
        return capped_sequence_time


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
        pos_array = np.array([pos.x, pos.y, pos.z])
        self.ball_pos = pos_array


    def calculate_min_paddle_vel(self, p0, pfinal, g, weights=None):
        if weights is None:
            weights = np.array([1.0, 1.0, 1.0])  # Default to equal weights


        def v0_norm_debug(T):
            if T <= 0:
                return np.inf  # Avoid division by zero
            # Compute v0 for given T
            v0 = (pfinal - p0 - 0.5 * g * (T**2)) / T
            # Weighted norm
            weighted_norm = np.sqrt(weights[0] * v0[0]**2 + weights[1] * v0[1]**2 + weights[2] * v0[2]**2)
            return weighted_norm

        # Initial guess for T
        T_guess = 1.0

        # Minimize ||v0|| with respect to T
        res = minimize(v0_norm_debug, x0=[T_guess], bounds=[(1e-3, None)])


        # Optimal time and velocity
        T_opt = res.x[0]
        min_paddle_vel = (pfinal - p0 - 0.5 * g * (T_opt**2)) / T_opt

        # Validate final position using kinematic equation
        p_computed = p0 + min_paddle_vel * T_opt + 0.5 * g * (T_opt**2)
        position_error = p_computed - pfinal

        # Debugging
        print("\n=== Results ===")
        print(f"Optimal Time of Flight (T): {T_opt}")
        print(f"Optimal Initial Velocity (v0): {min_paddle_vel}")
        print(f"Computed Final Position: {p_computed}")
        print(f"Target Final Position: {pfinal}")
        print(f"Position Error: {position_error}")

        return min_paddle_vel


    # TODO: REMOVE
    def test_find_min_Vi(self):
        p0 = self.ball_pos
        pfinal = self.bowl_pos
        g = np.array([0.0, 0.0, -1.0])

        T_opt, v0_opt = self.calculate_min_paddle_vel(p0, pfinal, g)
        print("Optimal time of flight:", T_opt)
        print("Optimal initial velocity:", v0_opt)
        print("Minimum initial speed:", np.linalg.norm(v0_opt))


    def ball_vel_callback(self, vel):
        self.ball_vel = p_from_Point(vel)

    def get_paddle_normal(self, R):
        return R[:, 1]


def main(args=None):
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 200, Trajectory)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
