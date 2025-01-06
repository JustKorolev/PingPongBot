import rclpy
import numpy as np
from scipy.optimize import minimize
from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from flyswatter.utils.GeneratorNode import GeneratorNode
from flyswatter.utils.TransformHelpers import *
from flyswatter.utils.TrajectoryUtils import *

# Grab the general fkin from HW6 P1.
from flyswatter.utils.KinematicChain import KinematicChain

# Import the format for the condition number message
from geometry_msgs.msg import Pose, Vector3, Point
from std_msgs.msg import Bool


#
# Controller Class
#
class Controller:
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
        self.tip_chain = KinematicChain(node, "world", "tip", self.jointnames())
        self.elbow_chain = KinematicChain(
            node, "world", "forearm_link", self.jointnames()[:-3]
        )
        self.shoulder_lift_chain = KinematicChain(
            node, "world", "upper_arm_link", self.jointnames()[:-4]
        )

        # BALL AND BOWL
        self.home_q_left = np.array([0, -pi / 2, pi / 2, pi / 2, 0, 0])
        self.home_q_right = np.array([pi, -pi / 2, pi / 2, pi / 2, 0, 0])

        # Initial position
        self.q0 = self.home_q_left
        self.home_p, self.home_R, _, _ = self.tip_chain.fkin(self.home_q_left)
        self.p0 = self.home_p
        self.R0 = self.home_R
        self.qdot0 = np.zeros(6)
        self.paddle_hit_vel = np.zeros(3)
        self.paddle_hit_normal = np.zeros(3)

        # Swing variables
        self.setup_time = 0
        self.hit_time = float("inf")
        self.return_time = float("inf")
        self.hit_pos = np.zeros(3)
        self.q_hit = np.zeros(6)
        self.qdot_hit = np.zeros(6)
        self.time_offset = 0

        # Kinematics variables
        self.qd = self.q0
        self.vd = np.zeros(3)
        self.wd = np.zeros(3)
        self.pd = self.p0
        self.Rd = self.R0
        self.nd = self.get_paddle_normal(self.R0)
        self.qddot = np.zeros(3)

        # Tuning constants
        self.lam = 20
        self.gamma = 0.23
        self.gamma_array = [self.gamma**2] * len(self.jointnames())
        self.max_joint_vels = np.array([2.5] * 6)
        self.max_vel_matrix = np.diag(self.max_joint_vels**2)

        # Robot parameters
        self.max_reach_rad = 1.3
        self.robot_base_position = np.array([0, 0, 0.1])

    def jointnames(self):
        return [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def evaluate(self, t, dt):

        # TODO: TESTING
        self.hit_pos = np.array([0.8, 0.8, 0.8])  # Robot reach radius is 1.3m
        self.ball_hit_velocity = np.zeros(3)
        ball_target_pos = np.array([2.0, -3.0, 0])
        g = np.array(
            [0.0, 0.0, -1.0]
        )  # Adjust magnitude as needed, e.g., -9.81 for real gravity

        # Set up sequence:
        # If ball is on right side, change starting position
        if self.hit_pos[1] < 0:
            self.setup_time = 2
        if t - self.time_offset < self.setup_time:
            # qd calculation using trajectory in joint space
            qd_hit, qddot_hit = spline(
                t - self.time_offset,
                self.setup_time,
                self.home_q_left,
                self.home_q_right,
                np.zeros(6),
                np.zeros(6),
            )

            # Calculating robot values using forward kinematics
            pd, Rd, Jv, Jw = self.tip_chain.fkin(qd_hit)
            vd = Jv @ qddot_hit
            wd = Jw @ qddot_hit
            nd = self.get_paddle_normal(Rd)

        # Hit sequence
        elif t - self.time_offset - self.setup_time < self.hit_time:

            # PRE CALCULATIONS FOR TRAJECTORY
            if t - self.time_offset - self.setup_time < dt:

                if np.linalg.norm(self.hit_pos) > self.max_reach_rad:
                    print("WARNING: OBJECT OUTSIDE OF WORKSPACE. ROBOT WILL NOT REACH.")

                # Calculate the required paddle velocity and normal to hit ball in basket
                self.paddle_hit_vel = self.calculate_min_paddle_vel(
                    self.hit_pos, ball_target_pos, g
                )
                self.paddle_hit_normal = self.paddle_hit_vel / np.linalg.norm(
                    self.paddle_hit_vel
                )

                print(f"Computed Paddle Hit Velocity: {self.paddle_hit_vel}")

                # Use newton raphson to converge to the final joint angles under task constraints
                self.q0 = self.qd
                self.q_hit = self.newton_raphson(
                    self.hit_pos, self.paddle_hit_normal, self.qd
                )
                _, _, Jvf, _ = self.tip_chain.fkin(self.q_hit)
                self.qdot_hit = np.linalg.pinv(Jvf) @ self.paddle_hit_vel

                # Calculate the trajectory time
                self.hit_time = self.calculate_sequence_time(self.q0, self.q_hit)
                self.return_time = self.hit_time

            if t - self.time_offset - self.setup_time > self.hit_time - dt:
                # Publishing at moment before impact
                p_actual, R_actual, Jv, _ = self.tip_chain.fkin(self.qd)
                v_actual = Jv @ self.qddot
                n_actual = self.get_paddle_normal(R_actual)
                self.tip_pose_msg = self.create_pose(p_actual, R_actual)
                self.tip_vel_msg = self.create_vel_vec(v_actual)
                self.tip_pose_pub.publish(self.tip_pose_msg)
                self.tip_vel_pub.publish(self.tip_vel_msg)

                # TODO: TROUBLESHOOTING VALUES AT IMPACT
                print(f"ACTUAL FINAL PADDLE VEL: {v_actual}")
                print(f"ACTUAL PADDLE NORMAL: {n_actual}")
                print(f"ACTUAL HIT POSITION: {p_actual}")
                print(f"ACTUAL JOINT POSITION: {self.qd}")
                print(f"DESIRED PADDLE VELOCITY: {self.paddle_hit_vel}")
                print(f"DESIRED PADDLE NORMAL: {self.paddle_hit_normal}")
                print(f"DESIRED HIT POSITION: {self.hit_pos}")
                print(f"DESIRED JOINT POSITION: {self.q_hit}")

            # qd calculation using trajectory in joint space
            qd_hit, qddot_hit = spline(
                t - self.time_offset - self.setup_time,
                self.hit_time,
                self.q0,
                self.q_hit,
                self.qdot0,
                self.qdot_hit,
            )

            # Calculating robot values using forward kinematics
            pd, Rd, Jv, Jw = self.tip_chain.fkin(qd_hit)
            vd = Jv @ qddot_hit
            wd = Jw @ qddot_hit
            nd = self.get_paddle_normal(Rd)

        # Return home sequence
        elif t - self.time_offset - self.setup_time - self.hit_time < self.return_time:
            qd_return, qddot_return = spline(
                t - self.time_offset - self.setup_time - self.hit_time,
                self.return_time,
                self.q_hit,
                self.home_q_left,
                self.qdot_hit,
                np.zeros(6),
            )
            pd, Rd, Jv, Jw = self.tip_chain.fkin(qd_return)
            vd = Jv @ qddot_return
            wd = Jw @ qddot_return
            nd = self.get_paddle_normal(Rd)

        else:
            return

        # Kinematics
        qdlast = self.qd
        pdlast = self.pd
        Rdlast = self.Rd
        ndlast = self.get_paddle_normal(Rdlast)
        pr, Rr, Jv, Jw = self.tip_chain.fkin(qdlast)
        nr = self.get_paddle_normal(Rr)

        # HIT SEQUENCE JACOBIAN CALCULATIONS
        if 0 < t - self.time_offset - self.setup_time < self.hit_time:

            # Position and normal errors
            error_p = ep(pdlast, pr)
            error_wd = np.cross(nr, ndlast)

            # Adjusted velocities
            adjusted_vd = vd + (self.lam * error_p)
            adjusted_wd_xz = self.adjusted_w(
                wd + (self.lam * error_wd), Rd
            )  # should be angular velocity
            combined_vwd = np.concatenate([adjusted_vd, adjusted_wd_xz])

            # Jacobian adjustments
            Jp = self.adjusted_jacobian(Jv, Jw, Rd)

        # SETUP / RETURN SEQUENCES
        else:
            # Position and normal errors
            error_p = ep(pdlast, pr)
            error_wd = eR(Rdlast, Rr)

            # Adjusted velocities
            adjusted_vd = vd + (self.lam * error_p)
            adjusted_wd = wd + (self.lam * error_wd)  # should be angular velocity
            combined_vwd = np.concatenate([adjusted_vd, adjusted_wd])

            # Jacobian adjustments
            Jp = np.vstack([Jv, Jw])

        # PRIMARY TASK: VELOCITY AND PADDLE ORIENTATION
        # Using targeted-removal / blending
        u, s, vT = np.linalg.svd(Jp, full_matrices=False)
        s_modified = np.diag(
            [(1 / s_i if s_i >= self.gamma else s_i / self.gamma**2) for s_i in s]
        )
        Jp_pinv = vT.T @ s_modified @ u.T

        # PRIMARY TASK: VELOCITY AND PADDLE ORIENTATION
        # SECONDARY TASK: REPULSION FROM FLOOR
        qsdot = self.repulsion(qdlast)
        N = (Jp_pinv @ Jp).shape[0]
        qddot = Jp_pinv @ combined_vwd + (np.eye(N) - Jp_pinv @ Jp) @ qsdot

        qd = qdlast + dt * qddot

        # Update state
        self.qd = qd
        self.pd = pd
        self.Rd = Rd
        self.vd = vd
        self.nd = nd
        self.wd = wd
        self.qddot = qddot

        return (qd, qddot, pd, vd, Rd, wd)

    def adjusted_jacobian(self, Jv_world, Jw_world, Rtip):
        Jw_tip = Rtip.T @ Jw_world
        Jw_tip_xz = np.vstack([Jw_tip[0, :], Jw_tip[2, :]])
        Jvw_adjusted = np.vstack([Jv_world, Jw_tip_xz])
        return Jvw_adjusted

    def adjusted_w(self, w_world, Rtip):
        w_tip = Rtip.T @ w_world
        w_tip_xz = np.array([w_tip[0], w_tip[2]])
        return w_tip_xz

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
        F_elbow = np.array([0, 0, 0.1 * np.e ** ((-elbow_distance_to_ground / 0.02))])
        F_tip_ground = np.array([0, 0, 0.1 * np.e ** (-tip_distance_to_ground / 0.02)])
        F_tip_base = (
            np.array(
                [
                    0.1 * tip_distance_to_base / tip_distance_to_base**2,
                    0.1 * tip_distance_to_base / tip_distance_to_base**2,
                    0.01 * tip_distance_to_base / tip_distance_to_base**2,
                ]
            )
            * tip_base_diff_directions
        )
        F_total = F_elbow + F_tip_ground + F_tip_base

        # Map the repulsion force acting at parm to the equivalent force
        # and torque actiing at the wrist point.
        T_elbow = np.cross(p_elbow - self.robot_base_position, F_elbow)
        T_tip_ground = np.cross(p_tip - self.robot_base_position, F_tip_ground)
        T_tip_base = np.cross(p_tip - self.robot_base_position, F_tip_base)
        T_total = T_elbow + T_tip_ground + T_tip_base

        J_stacked = np.vstack((Jv_elbow, Jw_elbow))
        force_torque_stacked = np.concatenate((F_total, T_total))

        # Convert the force/torque to joint torques (J^T).
        tau = J_stacked.T @ force_torque_stacked

        # Return the 3 joint torques as part of the 6 full joints.
        return np.concatenate([tau, np.zeros(3)])

    def calculate_shortest_angle(self, q0, qf):

        angle_diff = qf - q0
        wrapped_delta = fmod(angle_diff + pi, 2 * pi) - pi
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
            jac = self.adjusted_jacobian(Jv, Jw, R)
            p_error = pgoal - pr
            n_error = np.cross(nr, ngoal)
            n_error_xz = np.array([n_error[0], n_error[2]])
            combined_error = np.concatenate([p_error, n_error_xz])

            q = q + np.linalg.pinv(jac) @ (combined_error)

        # Unwrap q
        for i in range(len(q)):
            q[i] = fmod(q[i], 2 * pi)

            if shortest_angles:
                q[i] = self.calculate_shortest_angle(self.qd[i], q[i])

        print(f"DESIRED JOINT ANGLES: {q}")

        return q

    def calculate_sequence_time(self, q0, qf):
        max_sequence_time = 5
        q_diff_list = list(abs(qf - q0))
        max_q_diff = max(q_diff_list)
        sequence_time = (
            3 * (max_q_diff) ** 0.5 / self.max_joint_vels[q_diff_list.index(max_q_diff)]
        )
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
            weighted_norm = np.sqrt(
                weights[0] * v0[0] ** 2
                + weights[1] * v0[1] ** 2
                + weights[2] * v0[2] ** 2
            )
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
    generator = GeneratorNode("generator", 200, Controller)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
