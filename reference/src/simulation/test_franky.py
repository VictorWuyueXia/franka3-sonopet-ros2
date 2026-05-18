from franky import *

robot = Robot("172.16.0.2")

robot.relative_dynamics_factor = 0.05

motion = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)

robot.move(motion)