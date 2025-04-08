import numpy as np
import time


from robot_control.robot_arm import H1_2_ArmController
from robot_control.robot_arm_ik import H1_2_ArmIK
from robot_control.inspire_hand import H1HandController

from scipy.spatial.transform import Rotation as R

def rpy_xyz_to_matrix( x, y, z, roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw])
    rotation_matrix = r.as_matrix()

    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = [x, y, z]

    return matrix
def matrix_to_rpy_xyz(matrix):
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]

    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz')
    return *translation, roll, pitch, yaw, 

class ControllerWrapper():
    def __init__(self):
        self.arm_ctrl = H1_2_ArmController()
        self.finger_ctrl = H1HandController()
        self.arm_ik = H1_2_ArmIK(Visualization = False)
        self.arm_ctrl.speed_gradual_max()

    def set_fingers(self, left_hand_array, right_hand_array):
        self.finger_ctrl.crtl(right_hand_array, left_hand_array)
    def set_arms_pose(self, Lx, Ly, Lz, Lroll, Lpitch, Lyaw, Rx, Ry, Rz, Rroll, Rpitch, Ryaw):
        ###
        #ToDo:
        #  prevent hands from hitting body
        #  prevent other types of collisions
        #  add maximum velocity and torque limits
        #  Allow one arm control
        #  add emergency stop
        ###
        current_lr_arm_q  = self.arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = self.arm_ctrl.get_current_dual_arm_dq()

        constructed_l_target = rpy_xyz_to_matrix(Lx, Ly, Lz, Lroll, Lpitch, Lyaw)
        constructed_r_target = rpy_xyz_to_matrix(Rx, Ry, Rz, Rroll, Rpitch, Ryaw)

        sol_q, sol_tauff  = self.arm_ik.solve_ik(constructed_l_target, constructed_r_target, current_lr_arm_q, current_lr_arm_dq)
        self.arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)


if __name__ == '__main__':
    # arm
    controller = ControllerWrapper()
    try:
        i = 0
        while True:
            #print("\n\n\n")
            #controller.set_fingers(np.ones(6) * (i%2), np.ones(6) * ((i+1)%2))
            i += 1
            Lx, Ly, Lz, Lroll, Lpitch, Lyaw = 0.373 + (0.1*(i%10)), 0.323, 0.223, 0.0, 0.5899991834424116, 0.0
            Rx, Ry, Rz, Rroll, Rpitch, Ryaw = 0.373, -0.323, 0.223, 0.0, 0.0, 0.5899991834424116
            controller.set_arms_pose(Lx, Ly, Lz, Lroll, Lpitch, Lyaw, Rx, Ry, Rz, Rroll, Rpitch, Ryaw)
            time.sleep(1)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        #arm_ctrl.ctrl_dual_arm_go_home()
        print("Finally, exiting program...")
        exit(0)