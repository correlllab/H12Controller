import time
import numpy as np
import tkinter as tk

import pink
import qpsolvers
import pinocchio as pin

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.utils.thread import RecurrentThread

from robot_model import RobotModel
from channel_interface import CommandPublisher

class ArmController:
    def __init__(self, filename, dt, visualize=False):
        # initialize robot model
        self.robot_model = RobotModel(filename)
        self.dt = dt
        self.visualize = visualize

        if self.visualize:
            self.robot_model.init_visualizer()

        # initialize channel
        ChannelFactoryInitialize(id=0)

        # initialize subscriber in robot model
        self.robot_model.init_subscriber()
        time.sleep(0.5)
        self.robot_model.sync_subscriber()
        self.robot_model.update_kinematics()

        # initialize command publisher
        self.command_publisher = CommandPublisher()
        # enable upper body motors and set initial q & gain
        motor_ids = np.array([i for i in range(13, 27)])
        init_q = self.robot_model.q[np.array([i for i in range(13, 20)] + [i for i in range(32, 39)])]
        self.command_publisher.kp[13:27] = 150
        self.command_publisher.kd[13:27] = 3
        self.command_publisher.enable_motor(motor_ids, init_q)
        self.command_publisher.start_publisher()

        # initialize IK tasks
        # left arm end effector task
        self.left_ee_name = 'left_wrist_yaw_link'
        self.left_ee_task = pink.FrameTask(
            self.left_ee_name,
            position_cost=100.0,
            orientation_cost=30.0,
            lm_damping=0.1
        )
        # right arm end effector task
        self.right_ee_name = 'right_wrist_yaw_link'
        self.right_ee_task = pink.FrameTask(
            self.right_ee_name,
            position_cost=100.0,
            orientation_cost=30.0,
            lm_damping=0.1
        )
        # posture task as regularization
        self.posture_task = pink.PostureTask(
            cost=1e-3
        )

        # configuration trakcing robot states
        self.configuration = pink.Configuration(
            self.robot_model.model,
            self.robot_model.data,
            self.robot_model.zero_q,
        )

        # set initial target for all tasks
        self.tasks = [self.left_ee_task, self.right_ee_task, self.posture_task]
        for task in self.tasks:
            task.set_target_from_configuration(self.configuration)
        # select solver
        self.solver = qpsolvers.available_solvers[0]
        if 'osqp' in qpsolvers.available_solvers:
            self.solver = 'osqp'

    '''
    left end effector properties
    left_ee_transformation: transformation matrix of the left end effector
    left_ee_position: position of the left end effector
    left_ee_rotation: rotation matrix of the left end effector
    left_ee_rpy: roll, pitch, yaw of the left end effector
    left_ee_pose: pose of the left end effector (x, y, z, roll, pitch, yaw)
    '''
    @property
    def left_ee_transformation(self):
        return self.robot_model.get_frame_transformation(self.left_ee_name)

    @property
    def left_ee_position(self):
        return self.robot_model.get_frame_position(self.left_ee_name)

    @property
    def left_ee_rotation(self):
        return self.robot_model.get_frame_rotation(self.left_ee_name)

    @property
    def left_ee_rpy(self):
        return pin.rpy.matrixToRpy(self.left_ee_rotation)

    @property
    def left_ee_pose(self):
        return np.concatenate([self.left_ee_position,
                               self.left_ee_rpy])

    '''
    left end effector properties
    left_ee_transformation: transformation matrix of the left end effector
    left_ee_position: position of the left end effector
    left_ee_rotation: rotation matrix of the left end effector
    left_ee_rpy: roll, pitch, yaw of the left end effector
    left_ee_pose: pose of the left end effector (x, y, z, roll, pitch, yaw)
    '''
    @property
    def right_ee_transformation(self):
        return self.robot_model.get_frame_transformation(self.right_ee_name)

    @property
    def right_ee_position(self):
        return self.robot_model.get_frame_position(self.right_ee_name)

    @property
    def right_ee_rotation(self):
        return self.robot_model.get_frame_rotation(self.right_ee_name)

    @property
    def right_ee_rpy(self):
        return pin.rpy.matrixToRpy(self.right_ee_rotation)

    @property
    def right_ee_pose(self):
        return np.concatenate([self.right_ee_position,
                               self.right_ee_rpy])

    '''
    left end effector target properties
    left_ee_target_transformation: transformation matrix of the left end effector target
    left_ee_target_position: position of the left end effector target
    left_ee_target_rotation: rotation matrix of the left end effector target
    left_ee_target_rpy: roll, pitch, yaw of the left end effector target
    left_ee_target_pose: pose of the left end effector target (x, y, z, roll, pitch, yaw)
    '''
    @property
    def left_ee_target_transformation(self):
        return self.left_ee_task.transform_target_to_world.np

    @left_ee_target_transformation.setter
    def left_ee_target_transformation(self, transformation):
        assert(transformation.shape == (4, 4)), 'Transformation should be a 4x4 matrix.'
        self.left_ee_task.transform_target_to_world.np = transformation

    @property
    def left_ee_target_position(self):
        return self.left_ee_task.transform_target_to_world.translation

    @left_ee_target_position.setter
    def left_ee_target_position(self, position):
        assert(len(position) == 3), 'Position should be a list of 3 elements (x, y, z).'
        self.left_ee_task.transform_target_to_world.translation = np.array(position)

    @property
    def left_ee_target_rotation(self):
        return self.left_ee_task.transform_target_to_world.rotation

    @left_ee_target_rotation.setter
    def left_ee_target_rotation(self, rotation):
        assert(rotation.shape == (3, 3)), 'Rotation should be a 3x3 matrix.'
        self.left_ee_task.transform_target_to_world.rotation = rotation

    @property
    def left_ee_target_rpy(self):
        return pin.rpy.matrixToRpy(self.left_ee_target_rotation)

    @left_ee_target_rpy.setter
    def left_ee_target_rpy(self, rpy):
        assert(len(rpy) == 3), 'Rpy should be a list of 3 elements (roll, pitch, yaw).'
        self.left_ee_target_rotation = pin.rpy.rpyToMatrix(np.array(rpy))

    @property
    def left_ee_target_pose(self):
        return np.concatenate([self.left_ee_target_position,
                               self.left_ee_target_rpy])

    @left_ee_target_pose.setter
    def left_ee_target_pose(self, pose):
        assert(len(pose) == 6), 'Pose should be a list of 6 elements (x, y, z, roll, pitch, yaw).'
        self.left_ee_target_position = pose[:3]
        self.left_ee_target_rpy = pose[3:]

    '''
    right end effector target properties
    right_ee_target_transformation: transformation matrix of the right end effector target
    right_ee_target_position: position of the right end effector target
    right_ee_target_rotation: rotation matrix of the right end effector target
    right_ee_target_rpy: roll, pitch, yaw of the right end effector target
    right_ee_target_pose: pose of the right end effector target (x, y, z, roll, pitch, yaw)
    '''
    @property
    def right_ee_target_transformation(self):
        return self.right_ee_task.transform_target_to_world.np

    @right_ee_target_transformation.setter
    def right_ee_target_transformation(self, transformation):
        assert(transformation.shape == (4, 4)), 'Transformation should be a 4x4 matrix.'
        self.right_ee_task.transform_target_to_world.np = transformation

    @property
    def right_ee_target_position(self):
        return self.right_ee_task.transform_target_to_world.translation

    @right_ee_target_position.setter
    def right_ee_target_position(self, position):
        assert(len(position) == 3), 'Position should be a list of 3 elements (x, y, z).'
        self.right_ee_task.transform_target_to_world.translation = np.array(position)

    @property
    def right_ee_target_rotation(self):
        return self.right_ee_task.transform_target_to_world.rotation

    @right_ee_target_rotation.setter
    def right_ee_target_rotation(self, rotation):
        assert(rotation.shape == (3, 3)), 'Rotation should be a 3x3 matrix.'
        self.right_ee_task.transform_target_to_world.rotation = rotation

    @property
    def right_ee_target_rpy(self):
        return pin.rpy.matrixToRpy(self.right_ee_target_rotation)

    @right_ee_target_rpy.setter
    def right_ee_target_rpy(self, rpy):
        assert(len(rpy) == 3), 'Rpy should be a list of 3 elements (roll, pitch, yaw).'
        self.right_ee_target_rotation = pin.rpy.rpyToMatrix(np.array(rpy))

    @property
    def right_ee_target_pose(self):
        return np.concatenate([self.right_ee_target_position,
                               self.right_ee_target_rpy])

    @right_ee_target_pose.setter
    def right_ee_target_pose(self, pose):
        assert(len(pose) == 6), 'Pose should be a list of 6 elements (x, y, z, roll, pitch, yaw).'
        self.right_ee_target_position = pose[:3]
        self.right_ee_target_rpy = pose[3:]

    def control_loop(self):
        # sync robot model and compute forward kinematics
        self.robot_model.sync_subscriber()
        self.robot_model.update_kinematics()
        # update visualizer if needed
        if self.visualize:
            self.robot_model.update_visualizer()

        # update configuration and posture task
        self.configuration.update(self.robot_model.q)
        self.posture_task.set_target_from_configuration(self.configuration)

        # solve IK
        vel = pink.solve_ik(
            self.configuration,
            self.tasks,
            dt=self.dt,
            solver=self.solver,
            safety_break=False
        )

        # solve dynamics
        scaler = 3e-3
        tau = pin.rnea(self.robot_model.model,
                       self.robot_model.data,
                       self.robot_model.q + vel * scaler,
                       vel * scaler,
                       np.zeros(self.robot_model.model.nv))

        # send the velocity command to the robot
        self.command_publisher.q[13:20] = self.robot_model.q[13:20] + vel[13:20] * scaler
        self.command_publisher.q[20:27] = self.robot_model.q[32:39] + vel[32:39] * scaler
        self.command_publisher.tau[13:20] = tau[13:20]
        self.command_publisher.tau[20:27] = tau[32:39]

    def estop(self):
        self.command_publisher.estop()

if __name__ == '__main__':
    # Example usage
    arm_controller = ArmController('assets/h1_2/h1_2.urdf', dt=0.01, visualize=True)

    root = tk.Tk()
    root.title('Arm Controller')
    root.geometry('600x400')

    # pack sliders side by side
    left_frame = tk.Frame(root)
    right_frame = tk.Frame(root)  # Commented out for now
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10)  # Commented out for now

    # left hand sliders
    slider_lx = tk.Scale(left_frame, label="Left X",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_ly = tk.Scale(left_frame, label="Left Y",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lz = tk.Scale(left_frame, label="Left Z",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lr = tk.Scale(left_frame, label="Left Roll",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lp = tk.Scale(left_frame, label="Left Pitch",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lyaw = tk.Scale(left_frame, label="Left Yaw",
                           from_=-np.pi, to=np.pi, resolution=0.01,
                           orient=tk.HORIZONTAL, length=250)
    slider_lx.pack(in_=left_frame, pady=5)
    slider_ly.pack(in_=left_frame, pady=5)
    slider_lz.pack(in_=left_frame, pady=5)
    slider_lr.pack(in_=left_frame, pady=5)
    slider_lp.pack(in_=left_frame, pady=5)
    slider_lyaw.pack(in_=left_frame, pady=5)

    # right hand sliders
    slider_rx = tk.Scale(right_frame, label="Right X",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_ry = tk.Scale(right_frame, label="Right Y",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_rz = tk.Scale(right_frame, label="Right Z",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_rr = tk.Scale(right_frame, label="Right Roll",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_rp = tk.Scale(right_frame, label="Right Pitch",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_ryaw = tk.Scale(right_frame, label="Right Yaw",
                           from_=-np.pi, to=np.pi, resolution=0.01,
                           orient=tk.HORIZONTAL, length=250)
    slider_rx.pack(in_=right_frame, pady=5)
    slider_ry.pack(in_=right_frame, pady=5)
    slider_rz.pack(in_=right_frame, pady=5)
    slider_rr.pack(in_=right_frame, pady=5)
    slider_rp.pack(in_=right_frame, pady=5)
    slider_ryaw.pack(in_=right_frame, pady=5)

    # left hand target initialization
    left_ee_position = arm_controller.left_ee_target_pose[:3]
    slider_lx.set(left_ee_position[0])
    slider_ly.set(left_ee_position[1])
    slider_lz.set(left_ee_position[2])
    left_ee_rpy = arm_controller.left_ee_target_rpy
    slider_lr.set(left_ee_rpy[0])
    slider_lp.set(left_ee_rpy[1])
    slider_lyaw.set(left_ee_rpy[2])

    # Right hand target initialization
    right_ee_position = arm_controller.right_ee_target_pose[:3]
    slider_rx.set(right_ee_position[0])
    slider_ry.set(right_ee_position[1])
    slider_rz.set(right_ee_position[2])
    right_ee_rpy = arm_controller.right_ee_target_rpy
    slider_rr.set(right_ee_rpy[0])
    slider_rp.set(right_ee_rpy[1])
    slider_ryaw.set(right_ee_rpy[2])

    root.update()

    while True:
        root.update()
        # update left hand target
        lx = slider_lx.get()
        ly = slider_ly.get()
        lz = slider_lz.get()
        lr = slider_lr.get()
        lp = slider_lp.get()
        lyaw = slider_lyaw.get()
        arm_controller.left_ee_target_pose = [lx, ly, lz, lr, lp, lyaw]

        # update right hand target
        rx = slider_rx.get()
        ry = slider_ry.get()
        rz = slider_rz.get()
        rr = slider_rr.get()
        rp = slider_rp.get()
        ryaw = slider_ryaw.get()
        arm_controller.right_ee_target_pose = [rx, ry, rz, rr, rp, ryaw]

        arm_controller.control_loop()
        time.sleep(arm_controller.dt)
