import numpy as np
import time


from robot_control.robot_arm import H1_2_ArmController
from robot_control.robot_arm_ik import H1_2_ArmIK
from robot_control.inspire_hand import H1HandController

from scipy.spatial.transform import Rotation as R

from pose_constraints import start_left, start_right, in_safety_cylinder, out_of_range
import cv2
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

import threading
import matplotlib
matplotlib.use("Agg") 

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
        self.q = None
        self.tau = None
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
    
    def get_ee_pose(self, arm="L"):
        # 1. Get the configuration we want to inspect
        q = self.q

        # 2. Use ONE data object
        data = self.arm_ik.reduced_robot.data        # already exists

        # 3. Forward kinematics + frame placements
        pin.framesForwardKinematics(self.arm_ik.reduced_robot.model, data, q)
        pin.updateFramePlacements(self.arm_ik.reduced_robot.model, data)

        # 4. Pick the frame
        fid = self.arm_ik.reduced_robot.model.getFrameId("L_ee") if arm.upper().startswith("L") else self.arm_ik.reduced_robot.model.getFrameId("R_ee")
        # fid = self.arm_ik.reduced_robot.model.getFrameId("left_wrist_yaw_link") if arm.upper().startswith("L") else self.arm_ik.reduced_robot.model.getFrameId("R_ee")

        T_ref_EE = data.oMf[fid]                        # world → EE

        # 6. Decompose
        x, y, z = T_ref_EE.translation
        roll, pitch, yaw = R.from_matrix(T_ref_EE.rotation).as_euler('xyz', degrees=False)
        return x, y, z, roll, pitch, yaw

    def get_frame_wrench(
            self,
            arm="left",
            tau_meas=None,                      # raw motor torques, shape (nv,)
            strip_model=True,
            ref=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        """
        Cartesian wrench [Fx, Fy, Fz, Mx, My, Mz] acting on L_ee or R_ee.
        """
        # --------- 0. Resolve arm name -----------------------------------
        arm = arm.lower()[0]                   # 'left' → 'l', 'L' → 'l'
        frame_name = "L_ee" if arm == 'l' else "R_ee"

        # --------- 1. Acquire torques ------------------------------------
        # If the caller didn’t supply tau_meas, fall back to self.tau
        if tau_meas is None:
            tau_meas = self.tau                # <── NEW
        if strip_model and tau_meas is None:
            raise ValueError("tau_meas required when strip_model=True")

        model = self.arm_ik.reduced_robot.model
        data  = self.arm_ik.reduced_robot.data
        q     = self.q
        v     = getattr(self, "dq", np.zeros(model.nv))
        a     = np.zeros(model.nv)

        # Predicted gravity / inertia torques
        tau_model = pin.rnea(model, data, q, v, a) if strip_model else 0.0
        tau_ext   = tau_meas - tau_model

        # --------- 2. Spatial Jacobian -----------------------------------
        pin.computeJointJacobians(model, data, q)
        fid = model.getFrameId(frame_name)
        J   = pin.computeFrameJacobian(model, data, q, fid, ref)   # 6 × nv

        # --------- 3. Solve  Jᵀ w = τ_ext  -------------------------------
        wrench, *_ = np.linalg.lstsq(J.T, tau_ext, rcond=None)
        return wrench 
            

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
            self.q = sol_q
            self.tau = sol_tauff
            self.arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            #print(f"Step {i}/{steps}: Left Arm: {self.left_arm}, Right Arm: {self.right_arm}")
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.loop_dt - elapsed
            time.sleep(max(0, sleep_time))
        return
    def set_arms_velocity(self, vLx, vLy, vLz, vLroll, vLpitch, vLyaw, vRx, vRy, vRz, vRroll, vRpitch, vRyaw, total_time=1, blocking=True):
        if not blocking:
            t = threading.Thread(target=self.set_arms_velocity, args=(vLx, vLy, vLz, vLroll, vLpitch, vLyaw, vRx, vRy, vRz, vRroll, vRpitch, vRyaw, total_time, True), daemon=True)
            t.start()
            return t
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
            self.q = sol_q
            self.tau = sol_tauff
            self.arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            elapsed = time.perf_counter() - loop_start
            sleep_time = self.loop_dt - elapsed
            time.sleep(max(0, sleep_time))

    def set_arm_velocity(self, vx, vy, vz, vroll, vpitch, vyaw, arm="left", total_time=1, blocking=True):
        if arm == "left":
            self.set_arms_velocity(vx, vy, vz, vroll, vpitch, vyaw, 0,0,0,0,0,0, total_time=total_time, blocking=blocking)
        elif arm == "right":
            self.set_arms_velocity(0,0,0,0,0,0, vx, vy, vz, vroll, vpitch, vyaw, total_time=total_time, blocking=blocking)
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
    print(f"{controller.get_ee_pose('left')=}")

    
    i = 0
    #print("\n\n\n")
    controller.set_fingers(np.ones(6), np.ones(6))
    #Lx, Ly, Lz, Lroll, Lpitch, Lyaw = user_input, 0.2, 0, 0.0, 0.0, 0.0
    def nothing(x):
        pass

    # Create an OpenCV window with trackbars for 6 floating point inputs
    window_name = "ArmPositionControl"
    cv2.namedWindow(window_name)

    cv2.createTrackbar("X", window_name, 500, 1000, nothing)
    cv2.createTrackbar("Y", window_name, 700, 1000, nothing)
    cv2.createTrackbar("Z", window_name, 500, 1000, nothing)
    cv2.createTrackbar("Roll", window_name, 360, 720, nothing)
    cv2.createTrackbar("Pitch", window_name, 360, 720, nothing)
    cv2.createTrackbar("Yaw", window_name, 360, 720, nothing)
    cv2.createTrackbar("Fingers", window_name, 0, 100, nothing)
    cv2.createTrackbar("Arm", window_name, 0, 1, nothing)
    cv2.createTrackbar("Publish", window_name, 1, 1, nothing)
    while True:
        # Get values from the trackbars
        x = (cv2.getTrackbarPos("X", window_name)-500) / 1000.0
        y = (cv2.getTrackbarPos("Y", window_name)-500) / 1000.0
        z = (cv2.getTrackbarPos("Z", window_name)-500) / 1000.0
        roll = np.deg2rad(cv2.getTrackbarPos("Roll", window_name) - 360)
        pitch = np.deg2rad(cv2.getTrackbarPos("Pitch", window_name) - 360)
        yaw = np.deg2rad(cv2.getTrackbarPos("Yaw", window_name) - 360)
        fingers = cv2.getTrackbarPos("Fingers", window_name) / 100.0

        # Print the values for debugging
        # print(f"X: {x}, Y: {y}, Z: {z}, Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

        # Set the arm pose using the values
        if cv2.getTrackbarPos("Publish", window_name) == 1:
            arm = "left" if cv2.getTrackbarPos("Arm", window_name) == 0 else "right"
            pose = [x,y,z,roll, pitch, yaw]
            print(f"going to {arm} {pose=}")
            controller.set_arm_pose(x, y, z, roll, pitch, yaw, arm=arm)
            controller.set_fingers(np.ones(6) * fingers, np.ones(6) * fingers)
            print(f"{controller.get_ee_pose(arm=arm)=}")
            print(f"{controller.get_frame_wrench('left')=}")
        #controller.set_arm_velocity(x, y, z, roll, pitch, yaw, arm=arm, blocking=False)

        #print(f"{controller.get_ee_pose('left')=}")

        # controller.set_fingers(np.ones(6), np.ones(6))
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()