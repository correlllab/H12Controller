import numpy as np
import time


from robot_control.robot_arm import H1_2_ArmController
from robot_control.robot_arm_ik import H1_2_ArmIK
from robot_control.inspire_hand import H1HandController

from scipy.spatial.transform import Rotation as R

from pose_constraints import start_left, start_right, in_safety_cylinder, out_of_range
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
        self.arm_ik = H1_2_ArmIK(Visualization=False)

        self.left_arm = start_left  # [x, y, z, roll, pitch, yaw]
        self.right_arm = start_right
        self.max_velocity = 0.5  # meters per second
        self.step_hz = 50  # control loop frequency (Hz)
        self.loop_dt = 1.0 / self.step_hz

        self.set_arms_pose(*self.left_arm, *self.right_arm)
        self.set_fingers(np.ones(6), np.ones(6))

    def set_fingers(self, left_hand_array, right_hand_array):
        self.finger_ctrl.crtl(right_hand_array, left_hand_array)

    def n_steps(self, x, y, z, arm):
        current_position = self.left_arm[:3] if arm == "left" else self.right_arm[:3]
        target_position = np.array([x, y, z])
        distance = np.linalg.norm(target_position - current_position)
        dt = 1.0 / self.step_hz
        steps = int(np.ceil(distance / (self.max_velocity * dt)))
        return max(steps, 1)

    def set_arms_pose(self, Lx, Ly, Lz, Lroll, Lpitch, Lyaw, Rx, Ry, Rz, Rroll, Rpitch, Ryaw):
        if in_safety_cylinder(Lx, Ly, Lz):
            print(f"Left pose in safety cylinder {Lx=}, {Ly=}, {Lz=}")
            return
        if in_safety_cylinder(Rx, Ry, Rz):
            print(f"Right pose in safety cylinder {Rx=}, {Ry=}, {Rz=}")
            return
        if out_of_range(Lx, Ly, Lz, arm="left"):
            print(f"Left pose out of range {Lx=}, {Ly=}, {Lz=}")
            return
        if out_of_range(Rx, Ry, Rz, arm="right"):
            print(f"Right pose out of range {Rx=}, {Ry=}, {Rz=}")
            return

        steps = max(self.n_steps(Lx, Ly, Lz, "left"), self.n_steps(Rx, Ry, Rz, "right"))

        start_left_arm = self.left_arm.copy()
        start_right_arm = self.right_arm.copy()

        for i in range(steps + 1):
            loop_start = time.perf_counter()
            alpha = i / steps

            current_L = [
                start_left_arm[j] + alpha * (np.array([Lx, Ly, Lz, Lroll, Lpitch, Lyaw])[j] - start_left_arm[j])
                for j in range(6)
            ]
            current_R = [
                start_right_arm[j] + alpha * (np.array([Rx, Ry, Rz, Rroll, Rpitch, Ryaw])[j] - start_right_arm[j])
                for j in range(6)
            ]

            self.left_arm = current_L
            self.right_arm = current_R

            constructed_l_target = rpy_xyz_to_matrix(*self.left_arm)
            constructed_r_target = rpy_xyz_to_matrix(*self.right_arm)

            current_lr_arm_q = self.arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = self.arm_ctrl.get_current_dual_arm_dq()

            sol_q, sol_tauff = self.arm_ik.solve_ik(
                constructed_l_target,
                constructed_r_target,
                current_lr_arm_q,
                current_lr_arm_dq
            )

            if sol_q is None or sol_tauff is None:
                print(f"IK failed at step {i}")
                break

            self.arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            #print(f"Step {i}/{steps}: Left Arm: {self.left_arm}, Right Arm: {self.right_arm}")
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.loop_dt - elapsed
            time.sleep(max(0, sleep_time))
        return
    
    def set_arms_velocity(self, vLx, vLy, vLz, vLroll, vLpitch, vLyaw, vRx, vRy, vRz, vRroll, vRpitch, vRyaw, total_time=1):
        start_time = time.time()
        v_left = [vLx, vLy, vLz, vLroll, vLpitch, vLyaw]
        v_right = [vRx, vRy, vRz, vRroll, vRpitch, vRyaw]
        while time.time() - start_time < total_time:
            loop_start = time.perf_counter()
            current_L = [self.left_arm[j] + (v_left[j] * self.loop_dt) for j in range(len(self.left_arm)) ]
            current_R = [self.right_arm[j] + (v_right[j] * self.loop_dt) for j in range(len(self.right_arm)) ]
            if in_safety_cylinder(current_L[0], current_L[1], current_L[2]):
                print(f"Left pose in safety cylinder {current_L[0]=}, {current_L[1]=}, {current_L[2]=}")
                return
            if in_safety_cylinder(current_R[0], current_R[1], current_R[2]):
                print(f"Right pose in safety cylinder {current_R[0]=}, {current_R[1]=}, {current_R[2]=}")
                return
            if out_of_range(current_L[0], current_L[1], current_L[2], arm="left"):
                print(f"Left pose out of range {current_L[0]=}, {current_L[1]=}, {current_L[2]=}")
                return
            if out_of_range(current_R[0], current_R[1], current_R[2], arm="right"):
                print(f"Right pose out of range {current_R[0]=}, {current_R[1]=}, {current_R[2]=}")
            self.left_arm = current_L
            self.right_arm = current_R

            constructed_l_target = rpy_xyz_to_matrix(*self.left_arm)
            constructed_r_target = rpy_xyz_to_matrix(*self.right_arm)

            current_lr_arm_q = self.arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = self.arm_ctrl.get_current_dual_arm_dq()

            sol_q, sol_tauff = self.arm_ik.solve_ik(
                constructed_l_target,
                constructed_r_target,
                current_lr_arm_q,
                current_lr_arm_dq
            )

            if sol_q is None or sol_tauff is None:
                print(f"IK failed at step {i}")
                break

            self.arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            elapsed = time.perf_counter() - loop_start
            sleep_time = self.loop_dt - elapsed
            time.sleep(max(0, sleep_time))
    def set_arm_velocity(self, vx, vy, vz, vroll, vpitch, vyaw, arm="left", total_time=1):
        if arm == "left":
            self.set_arms_velocity(vx, vy, vz, vroll, vpitch, vyaw, 0,0,0,0,0,0, total_time=total_time)
        elif arm == "right":
            self.set_arms_velocity(0,0,0,0,0,0, vx, vy, vz, vroll, vpitch, vyaw, total_time=total_time)
        else:
            raise ValueError("Invalid arm specified. Use 'left' or 'right'.")
            

    def set_arm_pose(self, x, y, z, r, p, yaw, arm="left"):
        if in_safety_cylinder(x, y, z):
            print(f"{arm.capitalize()} pose in safety cylinder {x=}, {y=}, {z=}")
            return

        if arm == "left":
            self.set_arms_pose(*[x,y,z,r,p,yaw], *self.right_arm)
        elif arm == "right":
            self.set_arms_pose(*self.left_arm, *[x,y,z,r,p,yaw])
        else:
            raise ValueError("Invalid arm specified. Use 'left' or 'right'.")

        


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
            # print(f"X: {x}, Y: {y}, Z: {z}, Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

            # Set the arm pose using the values
            
            controller.set_arm_pose(x, y, z, roll, pitch, yaw, arm=arm)
            controller.set_fingers(np.array([1, 1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 1]))

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