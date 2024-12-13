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

        #BALL AND BOWL
        self.bowl_pos = np.array([1.5, 0.0, 0.0])
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
        self.hit_pos = np.zeros(3)
        self.hit_q = np.zeros(6)
        self.hit_qdot = np.zeros(6)
        self.time_offset = 0

        # Kinematics variables
        self.qd = self.q0
        self.vd = np.zeros(3)
        self.pd = self.p0
        self.Rd = self.R0
        self.nd = self.get_paddle_normal(self.R0)
        self.qddot = np.zeros(3)

        # Tuning constants
        self.lam = 10
        self.lam_second = 15
        self.gamma = 0.25
        self.gamma_array = [self.gamma ** 2] * len(self.jointnames())
        self.max_joint_vels = np.array([2, 2, 2, 2, 2, 2])
        self.joint_weights = np.ones(6) / self.max_joint_vels**2
        self.weight_matrix = np.diag(self.joint_weights)

        # Robot parameters
        self.max_reach_rad = 1.3



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
        self.hit_pos = np.array([0.5, 0.5, 0.5]) # Robot reach radius is 1.3m
        self.ball_hit_velocity = np.zeros(3)
        ball_target_pos = np.array([2, -2, 0])

        # Hit sequence
        if t - self.time_offset < self.hit_time:
            if t - self.time_offset < dt:

                if np.linalg.norm(self.hit_pos) > self.max_reach_rad:
                    print("WARNING: OBJECT OUTSIDE OF WORKSPACE. ROBOT WILL NOT REACH.")

                # Calculate the required paddle velocity and normal to hit ball in basket
                self.paddle_hit_vel = self.calculate_min_paddle_vel(self.hit_pos, ball_target_pos, -1.0)
                self.paddle_hit_normal = self.paddle_hit_vel / np.linalg.norm(self.paddle_hit_vel)

                # Use newton raphson to converge to the final joint angles under task constraints
                self.hit_q = self.newton_raphson(self.hit_pos, self.paddle_hit_normal, self.home_q)
                _, _, Jvf, _ = self.tip_chain.fkin(self.hit_q)
                self.hit_qdot = np.linalg.pinv(Jvf) @ self.paddle_hit_vel

                # Calculate the trajectory time
                self.hit_time = self.calculate_sequence_time(self.q0, self.hit_q)

            if t - self.time_offset > self.hit_time - dt:
                # Publishing at moment before impact
                self.tip_pose_msg = self.create_pose(self.pd, self.Rd)
                self.tip_vel_msg = self.create_vel_vec(self.vd)
                self.tip_pose_pub.publish(self.tip_pose_msg)
                self.tip_vel_pub.publish(self.tip_vel_msg)


                # TODO: TROUBLESHOOTING VALUES AT IMPACT
                print(f"ACTUAL FINAL PADDLE VEL: {self.vd}")
                print(f"ACTUAL PADDLE NORMAL: {self.nd}")
                print(f"ACTUAL HIT POSITION: {self.pd}")
                print(f"ACTUAL JOINT POSITION: {self.qd}")
                print(f"DESIRED PADDLE VELOCITY: {self.paddle_hit_vel}")
                print(f"DESIRED PADDLE NORMAL: {self.paddle_hit_normal}")
                print(f"DESIRED HIT POSITION: {self.hit_pos}")
                print(f"DESIRED JOINT POSITION: {self.hit_q}")

            # qd calculation using trajectory in joint space
            qd_hit, qddot_hit = spline(t - self.time_offset, self.hit_time, self.q0, self.hit_q, self.qdot0, self.hit_qdot)

            # Calculating robot values using forward kinematics
            pd, Rd, Jv, Jw = self.tip_chain.fkin(qd_hit)
            vd = Jv @ qddot_hit
            wd = Jw @ qddot_hit
            nd = self.get_paddle_normal(Rd)

        # Return home sequence
        # TODO: CHECK WHY ROBOT SOMETIMES DOESNT RETURN TO CORRECT HOME ANGLES
        elif t - self.time_offset < 2*self.hit_time:
            qd_return, qddot_return = spline(t - self.time_offset - self.hit_time, self.hit_time, self.hit_q, self.home_q, self.hit_qdot, np.zeros(6))
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

        # Position and normal errors
        error_p = ep(pdlast, pr)
        error_n = ep(ndlast, nr)

        # Adjusted velocities
        adjusted_vd = vd + ((4*self.lam * error_p) - (0.0 * error_n/dt))
        adjusted_nd = nd - ((2*self.lam * error_n) - (0.0 * error_n/dt))
        combined_vwd = np.concatenate([adjusted_vd, adjusted_nd])

        # Jacobian adjustments
        J_adjusted = self.adjusted_jacobian(Jv, Jw, nd)
        # J_p = J_adjusted[:3, :]
        # J_s = J_adjusted[3:, :]
        # J_pinv_p = np.linalg.pinv(J_p)
        # J_pinv_s = np.linalg.pinv(J_s)
        J_pinv = np.linalg.pinv(J_adjusted)

        # Primary task
        # qddot_main = J_pinv_p @ adjusted_vd

        # Secondary task
        # qddot_secondary = J_pinv_s @ adjusted_wd
        N = J_adjusted.shape[1]

        # BASIC QDDOT CALCULATION
        # TODO: CONSIDER USING TARGETED-REMOVAL/BLENDING
        jac_winv = np.linalg.pinv(J_adjusted.T @ J_adjusted +\
                                np.diag(self.gamma_array)) @ J_adjusted.T
        qddot = jac_winv @ combined_vwd

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

        # print(f"PADDLE POS: {self.pd}")
        # print(f"ACTUAL FINAL PADDLE VEL: {self.vd}")
        # print(f"ACTUAL PADDLE NORMAL: {self.nd}")
        # print(f"DESIRED PADDLE VELOCITY: {self.paddle_hit_vel}")
        # print(f"DESIRED PADDLE NORMAL: {self.paddle_hit_vel/np.linalg.norm(self.paddle_hit_vel)}")

        return (qd, qddot, pd, vd, Rd, wd)


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
        q_diff_list = list(qf - q0)
        max_q_diff = max(q_diff_list)
        return 4*(max_q_diff)**0.5 / self.max_joint_vels[q_diff_list.index(max_q_diff)]


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
        """
        Calculate the minimum paddle velocity needed to hit the ball to the target position.
        Args:
        - p0 (array): Initial position of the ball [x, y, z].
        - pfinal (array): Target final position [x, y, z].
        - g (array): Gravity vector [gx, gy, gz].
        - weights (array, optional): Weights for [x, y, z] components. Defaults to equal weights.

        Returns:
        - min_paddle_vel (array): Minimum initial velocity vector for the paddle [vx, vy, vz].
        """

        if weights is None:
            weights = np.array([1.0, 1.0, 1.0])  # Default to equal weights

        def v0_norm_debug(T):
            """
            Objective function to minimize weighted ||v0||.
            Includes debug information for intermediate velocity and position.
            """
            if T <= 0:
                return np.inf  # Avoid division by zero
            # Compute v0 for given T
            v0 = (pfinal - p0 - 0.5 * g * (T**2)) / T
            # Compute weighted norm
            weighted_norm = np.sqrt(weights[0] * v0[0]**2 + weights[1] * v0[1]**2 + weights[2] * v0[2]**2)

            # Compute the trajectory position at T
            p_computed = p0 + v0 * T + 0.5 * g * (T**2)

            # Debugging
            print(f"T: {T}, v0: {v0}, ||v0||: {np.linalg.norm(v0)}, Computed Final Position: {p_computed}")
            return weighted_norm

        # Initial guess for T
        T_guess = 1.0

        # Minimize ||v0|| with respect to T
        res = minimize(v0_norm_debug, x0=[T_guess], bounds=[(1e-3, None)])

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        # Optimal time and velocity
        T_opt = res.x[0]
        min_paddle_vel = (pfinal - p0 - 0.5 * g * (T_opt**2)) / T_opt

        # Validate final position using kinematic equation
        p_computed = p0 + min_paddle_vel * T_opt + 0.5 * g * (T_opt**2)
        position_error = np.linalg.norm(p_computed - pfinal)

        print("\n=== Final Results ===")
        print(f"Optimal Time of Flight (T): {T_opt}")
        print(f"Optimal Initial Velocity (v0): {min_paddle_vel}")
        print(f"Computed Final Position: {p_computed}")
        print(f"Target Final Position: {pfinal}")
        print(f"Position Error: {position_error}")

        if position_error > 1e-3:
            print("Warning: Significant position error detected! Adjust weights or verify dynamics.")

        return min_paddle_vel



    # TODO: REMOVE
    def test_find_min_Vi(self):
        p0 = self.ball_pos
        pfinal = self.bowl_pos
        g = np.array([0.0, 0.0, -9.81])

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
