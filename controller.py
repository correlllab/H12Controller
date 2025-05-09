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

    @property
    def left_ee_transformation(self):
        return self.robot_model.get_body_transformation(self.left_ee_name)

    @property
    def left_ee_position(self):
        return self.robot_model.get_body_position(self.left_ee_name)

    @property
    def left_ee_rotation(self):
        return self.robot_model.get_body_rotation(self.left_ee_name)

    @property
    def left_ee_pose(self):
        return np.concatenate((self.left_ee_position, pin.rpy.matrixToRpy(self.left_ee_rotation)))

    @property
    def right_ee_transformation(self):
        return self.robot_model.get_body_transformation(self.right_ee_name)

    @property
    def right_ee_position(self):
        return self.robot_model.get_body_position(self.right_ee_name)

    @property
    def right_ee_rotation(self):
        return self.robot_model.get_body_rotation(self.right_ee_name)

    @property
    def right_ee_pose(self):
        return np.concatenate([self.right_ee_position,
                               pin.rpy.matrixToRpy(self.right_ee_rotation)])

    @property
    def left_ee_target_pose(self):
        return np.concatenate((self.left_ee_task.transform_target_to_world.translation,
                               pin.rpy.matrixToRpy(self.left_ee_task.transform_target_to_world.rotation)))

    @left_ee_target_pose.setter
    def left_ee_target_pose(self, pose):
        assert(len(pose) == 6), 'Pose should be a list of 6 elements (x, y, z, roll, pitch, yaw).'
        self.left_ee_task.transform_target_to_world.translation = np.array(pose[:3])
        self.left_ee_task.transform_target_to_world.rotation = pin.rpy.rpyToMatrix(np.array(pose[3:]))

    @property
    def left_ee_target_position(self):
        return self.left_ee_task.transform_target_to_world.translation

    @left_ee_target_position.setter
    def left_ee_target_position(self, position):
        assert(len(position) == 3), 'Position should be a list of 3 elements (x, y, z).'
        self.left_ee_task.transform_target_to_world.translation = np.array(position)

    @property
    def right_ee_targe_pose(self):
        return np.concatenate((self.right_ee_task.transform_target_to_world.translation,
                               pin.rpy.matrixToRpy(self.right_ee_task.transform_target_to_world.rotation)))

    @right_ee_targe_pose.setter
    def right_ee_targe_pose(self, pose):
        assert(len(pose) == 6), 'Pose should be a list of 6 elements (x, y, z, roll, pitch, yaw).'
        self.right_ee_task.transform_target_to_world.translation = np.array(pose[:3])
        self.right_ee_task.transform_target_to_world.rotation = pin.rpy.rpyToMatrix(np.array(pose[3:]))

    @property
    def right_ee_target_position(self):
        return self.right_ee_task.transform_target_to_world.translation

    @right_ee_target_position.setter
    def right_ee_target_position(self, position):
        assert(len(position) == 3), 'Position should be a list of 3 elements (x, y, z).'
        self.right_ee_task.transform_target_to_world.translation = np.array(position)

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
    root.geometry('300x200')

    slider_x = tk.Scale(root, label="X",
                    from_=-1.0, to=1.0, resolution=0.01,
                    orient=tk.HORIZONTAL, length=250)
    slider_y = tk.Scale(root, label="Y",
                        from_=-1.0, to=1.0, resolution=0.01,
                        orient=tk.HORIZONTAL, length=250)
    slider_z = tk.Scale(root, label="Z",
                        from_=-1.0, to=1.0, resolution=0.01,
                        orient=tk.HORIZONTAL, length=250)
    slider_x.pack(pady=5)
    slider_y.pack(pady=5)
    slider_z.pack(pady=5)

    left_ee_position = arm_controller.left_ee_target_pose[:3]
    slider_x.set(left_ee_position[0])
    slider_y.set(left_ee_position[1])
    slider_z.set(left_ee_position[2])

    root.update()

    while True:
        root.update()
        x = slider_x.get()
        y = slider_y.get()
        z = slider_z.get()
        arm_controller.left_ee_target_position = [x, y, z]
        arm_controller.control_loop()
        time.sleep(arm_controller.dt)
