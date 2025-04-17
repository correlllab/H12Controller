import numpy as np
import time


from robot_control.robot_arm import H1_2_ArmController
from robot_control.robot_arm_ik import H1_2_ArmIK
from robot_control.inspire_hand import H1HandController

from scipy.spatial.transform import Rotation as R

from pose_constraints import start_left, start_right, in_safety_cylinder
import cv2

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
        self.left_arm = start_left
        self.right_arm = start_right
        self.set_arms_pose(*self.left_arm, *self.right_arm)
        self.set_fingers(np.ones(6), np.ones(6))

    def set_fingers(self, left_hand_array, right_hand_array):
        self.finger_ctrl.crtl(right_hand_array, left_hand_array)
    def set_arms_pose(self, Lx, Ly, Lz, Lroll, Lpitch, Lyaw, Rx, Ry, Rz, Rroll, Rpitch, Ryaw):
        ###
        #ToDo:
        #  prevent trajectory from hitting body
        #  add maximum velocity and torque limits
        #  add emergency stop
        ###

        # Check if the arms are in a safe position
        if in_safety_cylinder(Lx, Ly, Lz):
            print(f"Left pose In safety cylinder {Lx=}, {Ly=}, {Lz=}")
            return
        if in_safety_cylinder(Rx, Ry, Rz):
            print(f"Right pose In safety cylinder {Rx=}, {Ry=}, {Rz=}")
            return
        self.left_arm = [Lx, Ly, Lz, Lroll, Lpitch, Lyaw]
        self.right_arm = [Rx, Ry, Rz, Rroll, Rpitch, Ryaw]
        current_lr_arm_q  = self.arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = self.arm_ctrl.get_current_dual_arm_dq()

        constructed_l_target = rpy_xyz_to_matrix(*self.left_arm)
        constructed_r_target = rpy_xyz_to_matrix(*self.right_arm)

        sol_q, sol_tauff  = self.arm_ik.solve_ik(constructed_l_target, constructed_r_target, current_lr_arm_q, current_lr_arm_dq)
        self.arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
    def set_arm_pose(self, x,y,z,r,p,yaw, arm="left"):
        # Check if the arms are in a safe position
        if in_safety_cylinder(x, y, z):
            print(f"Pose In safety cylinder {x=}, {y=}, {z=}")
            return
        if arm == "left":
            self.left_arm = [x, y, z, r, p, yaw]
        elif arm == "right":
            self.right_arm = [x, y, z, r, p, yaw]
        else:
            raise ValueError("Invalid arm specified. Use 'left' or 'right'.")
        self.set_arms_pose(*self.left_arm, *self.right_arm)

if __name__ == '__main__':
    # arm
    controller = ControllerWrapper()
    arm = "left"
    try:
        i = 0
        #print("\n\n\n")
        controller.set_fingers(np.ones(6), np.ones(6))
        #Lx, Ly, Lz, Lroll, Lpitch, Lyaw = user_input, 0.2, 0, 0.0, 0.0, 0.0
        def nothing(x):
            pass

        # Create an OpenCV window with trackbars for 6 floating point inputs
        cv2.namedWindow("Arm Control")
        # Load and display an image from the assets folder
        image_path = "/home/humanoid/Programs/simple_arm_controller/assets/h1-2_tf.jpg"
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imshow("Arm Control", image)
        else:
            print(f"Failed to load image from {image_path}")

        cv2.createTrackbar("X", "Arm Control", 500, 1000, nothing)
        cv2.createTrackbar("Y", "Arm Control", 700, 1000, nothing)
        cv2.createTrackbar("Z", "Arm Control", 500, 1000, nothing)
        cv2.createTrackbar("Roll", "Arm Control", 0, 360, nothing)
        cv2.createTrackbar("Pitch", "Arm Control", 0, 360, nothing)
        cv2.createTrackbar("Yaw", "Arm Control", 0, 360, nothing)

        while True:
            # Get values from the trackbars
            x = (cv2.getTrackbarPos("X", "Arm Control")-500) / 1000.0
            y = (cv2.getTrackbarPos("Y", "Arm Control")-500) / 1000.0
            z = (cv2.getTrackbarPos("Z", "Arm Control")-500) / 1000.0
            roll = np.deg2rad(cv2.getTrackbarPos("Roll", "Arm Control"))
            pitch = np.deg2rad(cv2.getTrackbarPos("Pitch", "Arm Control"))
            yaw = np.deg2rad(cv2.getTrackbarPos("Yaw", "Arm Control"))

            # Print the values for debugging
            print(f"X: {x}, Y: {y}, Z: {z}, Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

            # Set the arm pose using the values
            
            controller.set_arm_pose(x, y, z, roll, pitch, yaw, arm=arm)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        #arm_ctrl.ctrl_dual_arm_go_home()
        print("Finally, exiting program...")
        exit(0)