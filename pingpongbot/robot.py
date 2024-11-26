import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from hw6code.KinematicChain     import KinematicChain
 # Import the format for the condition number message
from geometry_msgs.msg import Pose, Vector3

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
        self.hit_time = 2
        self.return_time = 2

        self.qd = self.q0
        self.pd = self.p0
        self.Rd = self.R0

        self.lam = 20
        self.lam_second = 10

        # Publishing
        self.tip_pose_pub = node.create_publisher(Pose, "/tip_pose", 10)
        self.tip_vel_pub = node.create_publisher(Vector3, "/tip_vel", 10)



    def jointnames(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def evaluate(self, t, dt):
        # TODO: change to whatever it should be dynamically
        # Desired hit position
        ball_p = np.array([0.8, 0.8, 0.8])
        # Desired swing back position
        swing_back_pd = ball_p - np.array([0, 1, 0.15])
        desired_hit_velocity = np.array([0, 1, 0])
        cycle_time = self.swing_back_time + self.hit_time + self.return_time
        t_cycle = fmod(t, cycle_time)

        # Swing back sequence
        if t_cycle < self.swing_back_time:
            pd, vd = spline(t_cycle, self.swing_back_time, self.home_p, swing_back_pd,
                            np.zeros(3), np.zeros(3))

        # Hit sequence
        elif t_cycle < self.swing_back_time + self.hit_time:
            pd, vd = spline(t_cycle - self.swing_back_time, self.hit_time,
                            swing_back_pd, ball_p, np.zeros(3), desired_hit_velocity)

        # Return home sequence
        elif t_cycle < self.swing_back_time + self.hit_time + self.return_time:
            pd, vd = spline(t_cycle - (self.swing_back_time + self.hit_time), self.return_time,
                            ball_p, self.home_p, desired_hit_velocity, np.zeros(3))
        


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
        combined_v_vec = np.concatenate((adjusted_vd, adjusted_wd))

        jac = np.vstack((Jv, Jw))
        jac_p_pinv = np.linalg.pinv(jac)

        N = jac.shape[1]
        qsdot = self.lam_second * (self.q_centers - qdlast)
        qddot = jac_p_pinv @ combined_v_vec +\
                    (np.eye(N) - jac_p_pinv @ jac) @ qsdot

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
        x, y, z = list(position)
        qx, qy, qz, qw = quat_from_R(orientation)
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose


    # Takes a numpy array velocity and returns a ROS vec3 message
    def create_vel_vec(self, velocity):
        vx, vy, vz = list(velocity)
        vec3 = Vector3(x=vx, y=vy, z=vz)
        return vec3




def main(args=None):
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
