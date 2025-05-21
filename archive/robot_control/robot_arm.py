import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_                                 # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
H1_2_Num_Motors = 35
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None



class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]
class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class H1_2_ArmController:
    def __init__(self):
        print("Initialize H1_2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        # initialize lowcmd publisher and lowstate subscriber
        ChannelFactoryInitialize(0)
        
        self.publisher = ChannelPublisher(kTopicLowCommand, LowCmd_)
        self.publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, LowState_)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.01)
            print("[H1_2_ArmController] Waiting to subscribe dds...")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        print(f"Current all body motor state q:\n{self.all_motor_q} \n")
        print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        print("Lock all joints except two arms...\n")

        arm_indices = set(member.value for member in H1_2_JointArmIndex)
        for id in H1_2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        print("Lock OK!\n")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        print("Initialize H1_2_ArmController OK!\n")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_2_LowState()
                for id in range(H1_2_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)
            #print("L2 norm of diff:", np.linalg.norm(arm_q_target-cliped_arm_q_target))

            for idx, id in enumerate(H1_2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.publisher.Write(self.msg)           

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_2_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        print("[H1_2_ArmController] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while True:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                print("[H1_2_ArmController] both arms have reached the home position.")
                break
            time.sleep(0.05)

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_2_JointIndex.kLeftAnkle.value,
            H1_2_JointIndex.kRightAnkle.value,
            # Left arm
            H1_2_JointIndex.kLeftShoulderPitch.value,
            H1_2_JointIndex.kLeftShoulderRoll.value,
            H1_2_JointIndex.kLeftShoulderYaw.value,
            H1_2_JointIndex.kLeftElbowPitch.value,
            # Right arm
            H1_2_JointIndex.kRightShoulderPitch.value,
            H1_2_JointIndex.kRightShoulderRoll.value,
            H1_2_JointIndex.kRightShoulderYaw.value,
            H1_2_JointIndex.kRightElbowPitch.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H1_2_JointIndex.kLeftElbowRoll.value,
            H1_2_JointIndex.kLeftWristPitch.value,
            H1_2_JointIndex.kLeftWristyaw.value,
            H1_2_JointIndex.kRightElbowRoll.value,
            H1_2_JointIndex.kRightWristPitch.value,
            H1_2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
    
class H1_2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

class H1_2_JointIndex(IntEnum):
    # Left leg
    kLeftHipYaw = 0
    kLeftHipRoll = 1
    kLeftHipPitch = 2
    kLeftKnee = 3
    kLeftAnkle = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipYaw = 6
    kRightHipRoll = 7
    kRightHipPitch = 8
    kRightKnee = 9
    kRightAnkle = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12

    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

    kNotUsedJoint0 = 27
    kNotUsedJoint1 = 28
    kNotUsedJoint2 = 29
    kNotUsedJoint3 = 30
    kNotUsedJoint4 = 31
    kNotUsedJoint5 = 32
    kNotUsedJoint6 = 33
    kNotUsedJoint7 = 34

if __name__ == "__main__":
    from robot_arm_ik import H1_2_ArmIK
    import pinocchio as pin 

    arm_ik = H1_2_ArmIK(Visualization = False)
    arm = H1_2_ArmController()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.2, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.2, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration
    q_target = np.zeros(35)
    tauff_target = np.zeros(35)

    user_input = input("Press anything to start \n")

    step = 0
    arm.speed_gradual_max()
    """
    while True:

        if step <= 120:
            angle = rotation_speed * step
            L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
            R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

            L_tf_target.translation += np.array([0.001,  0.001, 0.001])
            R_tf_target.translation += np.array([0.001, -0.001, 0.001])
        else:
            angle = rotation_speed * (240 - step)
            L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
            R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

            L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
            R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

        L_tf_target.rotation = L_quat.toRotationMatrix()
        R_tf_target.rotation = R_quat.toRotationMatrix()

        current_lr_arm_q  = arm.get_current_dual_arm_q()
        current_lr_arm_dq = arm.get_current_dual_arm_dq()
        l_target = L_tf_target.homogeneous
        r_target = R_tf_target.homogeneous
        
        sol_q, sol_tauff = arm_ik.solve_ik(l_target, r_target, current_lr_arm_q, current_lr_arm_dq)

        arm.ctrl_dual_arm(sol_q, sol_tauff)

        step += 1
        if step > 240:
            step = 0
        time.sleep(0.01)
    """
    from scipy.spatial.transform import Rotation as R
    def matrix_to_rpy_xyz(matrix):
        rotation_matrix = matrix[:3, :3]
        translation = matrix[:3, 3]

        r = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz')  # use degrees=False for radians

        return *translation, roll, pitch, yaw, 
    def rpy_xyz_to_matrix( x, y, z, roll, pitch, yaw):
        r = R.from_euler('xyz', [roll, pitch, yaw])
        rotation_matrix = r.as_matrix()

        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = [x, y, z]

        return matrix
    input("press anything to continue: ")
    while(True):
        current_lr_arm_q  = arm.get_current_dual_arm_q()
        current_lr_arm_dq = arm.get_current_dual_arm_dq()
        l_target = np.array([[ 0.83094,  0.,       0.55636 , 0.373 ],
                        [ 0.,       1. ,      0.     ,  0.323  ],
                        [-0.55636,  0. ,      0.83094 , 0.223  ],
                        [ 0.,       0. ,      0.   ,   1.     ]])
        Lx, Ly, Lz, Lroll, Lpitch, Lyaw = matrix_to_rpy_xyz(l_target)
        reconstructed_l_target = rpy_xyz_to_matrix(Lx, Ly, Lz, Lroll, Lpitch, Lyaw)
        assert np.allclose(l_target, reconstructed_l_target), "Reconstructed Ltarget does not match original matrix"
        print(f"l_target:\n{l_target}\n{matrix_to_rpy_xyz(l_target)}")
        

        r_target = np.array([[ 0.83094, -0.55636 , 0.   ,    0.373  ],
                            [ 0.55636,  0.83094 , 0.   ,   -0.323  ],
                            [ 0.     , 0.      , 1.    ,   0.223  ],
                            [ 0.     ,  0.     ,  0.   ,    1.     ]])
        Rx, Ry, Rz, Rroll, Rpitch, Ryaw = matrix_to_rpy_xyz(r_target)
        reconstructed_r_target = rpy_xyz_to_matrix(Rx, Ry, Rz, Rroll, Rpitch, Ryaw)
        assert np.allclose(r_target, reconstructed_r_target), "Reconstructed Rtarget does not match original matrix"
        print(f"r_target:\n{r_target}\n{matrix_to_rpy_xyz(r_target)}")


        sol_q, sol_tauff  = arm_ik.solve_ik(l_target, r_target, current_lr_arm_q, current_lr_arm_dq)
        arm.ctrl_dual_arm(sol_q, sol_tauff)

