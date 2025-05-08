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
        self.command_publisher.kp[13:20] = 150
        self.command_publisher.kd[13:20] = 3
        self.command_publisher.enable_motor(motor_ids, init_q)
        self.command_publisher.start_publisher()

        # initialize IK tasks
        # left arm end effector task
        self.left_ee_name = 'L_hand_base_link'
        self.left_ee_task = pink.FrameTask(
            self.left_ee_name,
            position_cost=50.0,
            orientation_cost=10.0,
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
            self.robot_model.q,
        )

        # set initial target for all tasks
        self.tasks = [self.left_ee_task, self.posture_task]
        for task in self.tasks:
            task.set_target_from_configuration(self.configuration)
        # select solver
        self.solver = qpsolvers.available_solvers[0]
        if 'osqp' in qpsolvers.available_solvers:
            self.solver = 'osqp'

    def set_left_arm_pose(self, pose):
        assert(len(pose) == 6), 'Pose should be a list of 6 elements (x, y, z, roll, pitch, yaw).'
        self.set_left_arm_position(pose[:3])
        self.set_left_arm_orientation(pose[3:])

    def set_left_arm_position(self, position):
        assert(len(position) == 3), 'Position should be a list of 3 elements (x, y, z).'
        self.left_ee_task.transform_target_to_world.translation = np.array(position)

    def set_left_arm_orientation(self, orientation):
        assert(len(orientation) == 3), 'Orientation should be a list of 3 elements (roll, pitch, yaw).'
        self.left_ee_task.transform_target_to_world.rotation = pin.rpy.rpyToMatrix(
            np.array(orientation)
        )

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
        tau = pin.rnea(self.robot_model.model,
                       self.robot_model.data,
                       self.robot_model.q + vel * 1e-3,
                       vel * 1e-3,
                       np.zeros(self.robot_model.model.nv))

        # send the velocity command to the robot
        self.command_publisher.q[13:20] = self.robot_model.q[13:20] + vel[13:20] * 1e-3
        self.command_publisher.q[20:27] = self.robot_model.q[32:39] + vel[32:39] * 1e-3
        self.command_publisher.tau[13:20] = tau[13:20]
        self.command_publisher.tau[20:27] = tau[32:39]

    def get_left_ee_transformation(self):
        return self.robot_model.get_body_transformation(self.left_ee_name)

    def get_left_ee_position(self):
        return self.robot_model.get_body_position(self.left_ee_name)

    def get_left_ee_orientation(self):
        return self.robot_model.get_body_orientation(self.left_ee_name)

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

    left_ee_position = arm_controller.get_left_ee_position()
    slider_x.set(left_ee_position[0])
    slider_y.set(left_ee_position[1])
    slider_z.set(left_ee_position[2])

    root.update()

    while True:
        root.update()
        x = slider_x.get()
        y = slider_y.get()
        z = slider_z.get()
        arm_controller.set_left_arm_position([x, y, z])
        arm_controller.control_loop()
        time.sleep(arm_controller.dt)
