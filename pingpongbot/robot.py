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
from std_msgs.msg import Bool

#
# Trajectory Class
#
class Trajectory():
    # Initialization
    def __init__(self, node):

        # Publishing
        self.tip_pose_pub = node.create_publisher(Pose, "/tip_pose", 100)
        self.tip_vel_pub = node.create_publisher(Vector3, "/tip_vel", 100)
        self.start_pub = node.create_publisher(Bool, "/start", 10)
        self.start_pub.publish(Bool())

        # Subscribing
        node.create_subscription(Point, "/ball_pos", self.ball_pos_callback, 10)
        node.create_subscription(Point, "/ball_vel", self.ball_vel_callback, 10)

        # Subscription variables
        # Store ball positions and times in a list
        #We can modify this to whatever we desire, i think 0.5 makese sense
        self.ball_pos = np.ones(3)
        self.ball_vel = np.zeros(3)

        self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())

        self.home_q = np.array([-0.1, -2.2, 2.4, -1.0, 1.6, 1.6])
        self.q_centers = np.array([-pi/2, -pi/2, 0, -pi/2, 0, 0])

        # Initial position
        self.q0 = self.home_q
        self.home_p, self.home_R, _, _  = self.chain.fkin(self.home_q)
        self.p0 = self.home_p
        self.R0 = self.home_R
        self.qdot0 = np.zeros(6)

        # Swing variables
        self.hit_z = 0.1
        self.hit_time = float("inf")
        self.ball_hit_velocity = np.zeros(3)
        self.hit_pos = np.zeros(3)
        self.update_hit_parameters(self.hit_z)
        self.return_time = float("inf")
        self.hit_q = np.zeros(6)
        self.return_q = np.zeros(6)
        self.time_offset = 0

        self.qd = self.q0
        self.vd = np.zeros(3)
        self.pd = self.p0
        self.Rd = self.R0
        self.nd = self.get_paddle_normal(self.R0)

        # Tuning constants
        self.lam = 10
        self.lam_second = 15
        self.gamma = 0
        self.gamma_array = [self.gamma ** 2] * len(self.jointnames())
        self.joint_weights = np.array([1, 1, 1, 1, 1, 1])
        self.weight_matrix = np.diag(self.joint_weights)



    def jointnames(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def get_target_surface_normal(self):
    #Target normal is along the z-axis in the world frame? Double check this
        return nz()

    def get_current_tool_normal(self):
        # get curr orientation matrix from forward kines
        _, R_current, _, _ = self.chain.fkin(self.qd)
        return R_current[:, 2]  #z-axis of the current rotation matrix

    def compute_rotation_from_normals(self, current_normal, target_normal):
        current_normal = current_normal / np.linalg.norm(current_normal)
        target_normal = target_normal / np.linalg.norm(target_normal)

        # Compute the rotation axis and angle
        rotation_axis = np.cross(current_normal, target_normal)
        rotation_angle = np.arccos(np.clip(np.dot(current_normal, target_normal), -1.0, 1.0))

        # Handle edge cases (normals are parallel or anti-parallel)
        if np.linalg.norm(rotation_axis) < 1e-6:
            # Parallel normals
            return Reye()
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # Normalize the axis

        # Use Rodrigues formula to compute the rotation matrix
        return rodrigues_formula(rotation_axis, rotation_angle)



    # def adjust_jacobian(self, Jv, Jw):
    #     J_combined = np.vstack((Jv, Jw))

    #     # get the Surface Normals
    #     target_normal = self.get_target_surface_normal()
    #     current_normal = self.get_current_tool_normal()
    #     #Rotation Matrix to Align the Normals
    #     R_o = self.compute_rotation_from_normals(current_normal, target_normal)
    #     R_full = np.block([
    #         [R_o,           np.zeros((3, 3))],
    #         [np.zeros((3, 3)),            R_o]])
    #     # Rotate the Jacobian
    #     J_rotated = R_full @ J_combined
    #     # Remove the last row of the Jacobian
    #     J_adjusted = J_rotated[:-1, :]

    #     return J_adjusted

    def adjusted_jacobian(self, Jv, Jw, nr):
        n_cross = crossmat(nr)
        Jn = n_cross @ Jw
        J_adjusted = np.vstack([Jv, Jn])
        return J_adjusted



    def evaluate(self, t, dt):
        pd = self.pd
        desired_ball_velocity = np.array([0, 0, 2])
        paddle_hit_vel = 0.5 * (desired_ball_velocity - self.ball_hit_velocity)
        paddle_normal = paddle_hit_vel / np.linalg.norm(paddle_hit_vel)
        print(f"Paddle hit vel: {paddle_hit_vel}")
        print(f"Paddle normal: {paddle_normal}")

        _, _, Jvf, _ = self.chain.fkin(self.hit_q)
        qdotf = np.linalg.pinv(Jvf) @ paddle_hit_vel

        # Hit sequence
        if t - self.time_offset < self.hit_time:
            if t < dt: # TODO:
                # print("self.hit_rotation after assignment:\n", self.hit_rotation)

                self.hit_q = self.newton_raphson(self.hit_pos, paddle_normal, self.home_q)


            # qd calculation
            qd_hit, qddot_hit = spline(t - self.time_offset, self.hit_time, self.q0, self.hit_q, self.qdot0, qdotf)

            pd, Rd, Jv, Jw = self.chain.fkin(qd_hit)
            vd = Jv @ qddot_hit
            wd = Jw @ qddot_hit
            nd = self.get_paddle_normal(Rd)

            # Post hit
            if t - self.time_offset > self.hit_time - dt:
                self.time_offset += self.hit_time
                self.update_hit_parameters(self.hit_z)
                self.q0 = self.hit_q
                self.qdot0 = qdotf
                print(f"ACTUAL FINAL HIT VEL: {self.vd}")
                print(f"ACTUAL PADDLE NORMAL: {self.nd}")
                return 







        # TODO: TEMPORARY UNCHANGING ROTATION -- CHANGE
        # Kinematics
        qdlast = self.qd
        pdlast = self.pd
        Rdlast = self.Rd
        ndlast = self.get_paddle_normal(Rdlast)
        pr, Rr, Jv, Jw = self.chain.fkin(qdlast)
        nr = self.get_paddle_normal(Rr)


        #print("Desired Rotation Matrix:\n", Rd)

        # Position and rotation errors
        error_p = ep(pdlast, pr)
        error_n = ep(ndlast, nr)

        # Adjusted velocities
        adjusted_vd = vd + (self.lam * error_p)
        adjusted_nd = nd + (self.lam * error_n)
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
        # qddot = self.weight_matrix @ jac_winv.T @\
        #     np.linalg.pinv(jac_winv @ self.weight_matrix @ jac_winv.T) @ combined_vwd

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
        self.vd = vd

        # Publishing
        self.tip_pose_msg = self.create_pose(self.pd, self.Rd)
        self.tip_vel_msg = self.create_vel_vec(self.vd)
        self.tip_pose_pub.publish(self.tip_pose_msg)
        self.tip_vel_pub.publish(self.tip_vel_msg)

        return (qd, qddot, pd, vd, Rd, wd)

    # Newton Raphson
    def newton_raphson(self, pgoal, ngoal, q0):
        # Number of steps to try.
        N = 100

        # Setting initial q
        q = q0

        for _ in range(N):
            (pr, R, Jv, Jw) = self.chain.fkin(q)
            nr = self.get_paddle_normal(R)
            jac = self.adjusted_jacobian(Jv, Jw, nr)
            p_error = pgoal - pr
            n_error = ngoal - nr
            combined_error = np.concatenate([p_error, n_error])

            q = q + np.linalg.pinv(jac) @ (combined_error)

        # Unwrap q
        for i in range(len(q)):
            q[i] = fmod(q[i], 2*pi)

        # # Adjust q desired to closest joint value
        q_closest = np.zeros(len(q))

        for i in range(len(q)):
            q_closest[i] = min([q[i] + 2*pi*k for k in range(-4, 4)],
                                key=lambda angle: abs(self.qd[i] - angle))

        print(q_closest)
        return q_closest


    # TODO: MAKE THIS MORE SOPHISTICATED
    def calculate_sequence_time(self, q0, qf, qddot0, qddotf):
        # TODO: THIS IS VERY JANK
        # avg_qddot = np.linalg.norm(qddotf - qddot0) / 4
        # print(np.linalg.norm((qf - q0)) / avg_qddot)
        return np.linalg.norm((qf - q0)) / 3 # TODO NEEDS WORK


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


    def update_hit_parameters(self, z_target):
        p0 = self.ball_pos
        x0, y0, z0 = list(p0)
        v0 = self.ball_vel
        vx0, vy0, vz0 = v0[0], v0[1], v0[2]

        # Kinematic parameters
        g = -1
        a = 0.5 * g
        b = vz0
        c = (z0 - z_target)

        discriminant = b**2 - 4*a*c

        t_sol1 = (-b + np.sqrt(discriminant)) / (2*a)
        t_sol2 = (-b - np.sqrt(discriminant)) / (2*a)

        # Choose positive time solution
        t_candidates = [t for t in [t_sol1, t_sol2] if t > 0]
        if not t_candidates:
            return None
        t_hit = min(t_candidates)

        # Compute the hit position
        x_hit = x0 + vx0 * t_hit
        y_hit = y0 + vy0 * t_hit

        # Compute the velocity at t_hit:
        # vx and vy remain constant, vz changes due to gravity
        v_hit_z = vz0 + g * t_hit

        self.ball_hit_velocity = np.array([vx0, vy0, v_hit_z])
        self.hit_time = t_hit
        self.hit_pos = np.array([x_hit, y_hit, self.hit_z])

        return (t_hit, x_hit, y_hit, self.ball_hit_velocity)



    def ball_vel_callback(self, vel):
        self.ball_vel = p_from_Point(vel)

    def get_paddle_normal(self, R):
        return R[:, 1]


def main(args=None):
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 300, Trajectory)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
