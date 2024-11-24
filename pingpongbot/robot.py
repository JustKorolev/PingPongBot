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
        self.chain = KinematicChain(node, 'world', 'tool0', self.jointnames())

        self.q0 = np.radians(np.array([0, 0, 0, 0, 0, 0]))
        self.p0 = np.array([0.0, 0.0, 0.0])
        self.R0 = Reye()

        self.pleft = np.array([0.3, 0.5, 0.15])
        self.pright = np.array([-0.3, 0.5, 0.15])
        self.Rleft = Rotx(-np.pi / 2) @ Roty(-np.pi / 2)
        self.Rright = Reye()

        self.qd = self.q0
        self.pd = self.p0
        self.Rd = self.R0

        self.lam = 20


    def jointnames(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def evaluate(self, t, dt):
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

        # qdlast = self.qd
        # pdlast = self.pd
        # Rdlast = self.Rd

        # (p, R, Jv, Jw) = self.chain.fkin(qdlast)

        # J = np.vstack((Jv, Jw))

        # vr = vd + self.lam * ep(pdlast, p)
        # wr = wd + self.lam * eR(Rdlast, R)
        # xrdot = np.concatenate((vr, wr))
        # qddot = np.linalg.pinv(J) @ xrdot

        # qd = qdlast + dt * qddot
        Rd = Reye()
        wd = np.zeros(3)
        qd = self.q0
        qddot = np.zeros(6)
        pd = self.p0
        vd = np.zeros(3)

        self.qd = qd
        self.pd = pd
        self.Rd = Rd

        return qd, qddot, pd, vd, Rd, wd

def main(args=None):
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)
    generator.spin()
    generator.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
