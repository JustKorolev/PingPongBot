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
from std_msgs.msg import Float64

#
# Trajectory Class
#
class Trajectory():
    # Initialization
    def __init__(self, node):
        self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())
        
        # Our home joint angles
        self.home_q = np.array([-0.1, -2.2, 2.4, -1.0, 1.6, 1.6])

        # Initial position
        self.q0 = self.home_q
        self.p0 = self.home_p
        self.R0 = self.home_R
        (self.home_p, self.home_R, _, _)  = self.chain.fkin(self.q0)


        self.qd = self.q0
        self.pd = self.p0
        self.Rd = self.R0

        self.lam = 0


    def jointnames(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def evaluate(self, t, dt):

        # CONSTANTS
        swing_back_time = 4.0

        # Desired hit position
        pd_f = np.array([0.8, 0.8, 0.8])
        swing_back_pd = pd_f - np.array([0, 0.5, 0.15])

        # Swing back sequence
        if t < swing_back_pd:
            pd, vd = spline(t, swing_back_time, self.home_p, swing_back_pd,
                            np.zeros(3), np.zeros(3))

            Rd = self.R0
            wd = np.zeros(3)
        else:
            pass




        # if t < 3.0:
        #     s = t / 3.0
        #     pd = self.p0 + (self.pright - self.p0) * s
        #     vd = (self.pright - self.p0) / 3.0

        #     Rd = Reye()
        #     wd = np.zeros(3)
        # else:
        #     T = 5.0
        #     t0 = 3.0

        #     s = np.cos(2 * np.pi * (t - t0) / T)
        #     sdot = -2 * np.pi / T * np.sin(2 * np.pi * (t - t0) / T)

        #     pd = np.array([-0.3 * s, 0.5, 0.5 - 0.35 * s ** 2])
        #     vd = np.array([-0.3 * sdot, 0.0, -0.7 * s * sdot])

        #     alpha = np.pi / 4 * (1 - s)
        #     alphadot = -np.pi / 4 * sdot
        #     Rd = Rotx(-alpha) @ Roty(-alpha)
        #     wd = -nx() * alphadot - Rotx(-alpha) @ ny() * alphadot

        qdlast = self.qd
        pdlast = self.pd
        Rdlast = self.Rd

        (p, R, Jv, Jw) = self.chain.fkin(qdlast)

        J = np.vstack((Jv, Jw))

        vr = vd + self.lam * ep(pdlast, p)
        wr = wd + self.lam * eR(Rdlast, R)
        xrdot = np.concatenate((vr, wr))
        qddot = np.linalg.pinv(J) @ xrdot

        qd = qdlast + dt * qddot

        self.qd = qd
        self.pd = pd
        self.Rd = Rd

        return (qd, qddot, pd, vd, Rd, wd)

def main(args=None):
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
