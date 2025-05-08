import os
import numpy as np

import time
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import pink
import qpsolvers

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from channel_interface import StateSubscriber

class RobotModel:
    def __init__(self, filename: str):
        # break file name
        ext = os.path.splitext(filename)[1]
        dirs = os.path.dirname(filename)
        # check file extension, load the model
        if ext == '.xml':
            self.model, self.collision_model, self.visual_model = pin.buildModelsFromMJCF(
                filename=filename
            )
        elif filename.endswith('.urdf'):
            self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
                filename=filename,
                package_dirs=dirs
            )
        else:
            raise ValueError('Unsupported file format. Please provide a .xml or .urdf file.')
        # intiialize data for the model
        self.data = self.model.createData()

        # field variabels tracking joint states
        self._q = np.zeros(self.model.nq)
        self._dq = np.zeros(self.model.nq)
        self._tau = np.zeros(self.model.nq)

        # initialize with zero joint positions
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    @property
    def q(self):
        return np.copy(self._q)

    @property
    def dq(self):
        return np.copy(self._dq)

    @property
    def get_tau(self):
        return np.copy(self._tau)

    def init_visualizer(self):
        try:
            self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model,
                                         copy_models=False, data=self.data)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel('unitree_h1_2')
        except ImportError as err:
            print('ImportError: MeshcatVisualizer requires the meshcat package.')
            print(err)
            exit(0)

    def init_subscriber(self):
        self.state_subscriber = StateSubscriber()

    def sync_subscriber(self):
        # update the q, dq, tau
        self._q[0:20] = self.state_subscriber.q[0:20]
        self._q[32:39] = self.state_subscriber.q[20:27]
        self._dq[0:20] = self.state_subscriber.dq[0:20]
        self._dq[32:39] = self.state_subscriber.dq[20:27]
        self._tau[0:20] = self.state_subscriber.get_tau[0:20]
        self._tau[32:39] = self.state_subscriber.get_tau[20:27]

    def forward_kinematics(self):
        # udpate data with the current joint positions
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def update_viz(self):
        self.viz.display()

    def get_body_transformation(self, body_name: str):
        body_id = self.model.getBodyId(body_name)
        transformation = self.data.oMf[body_id]
        return transformation.np

    def get_body_position(self, body_name: str):
        body_id = self.model.getBodyId(body_name)
        transformation = self.data.oMf[body_id]
        return transformation.translation

    def get_body_rotation(self, body_name: str):
        body_id = self.model.getBodyId(body_name)
        transformation = self.data.oMf[body_id]
        return transformation.rotation

    def get_joint_transformation(self, joint_name: str):
        joint_id = self.model.getJointId(joint_name)
        transformation = self.data.oMi[joint_id]
        return transformation.np

    def get_joint_position(self, joint_name: str):
        joint_id = self.model.getJointId(joint_name)
        transformation = self.data.oMi[joint_id]
        return transformation.translation

    def get_joint_rotation(self, joint_name: str):
        joint_id = self.model.getJointId(joint_name)
        transformation = self.data.oMi[joint_id]
        return transformation.rotation


if __name__ == '__main__':
    ChannelFactoryInitialize(id=0)
    # Example usage
    robot_model = RobotModel('assets/h1_2/h1_2.urdf')
    # robot_model = RobotModel('assets/h1_2/h1_2.xml')
    robot_model.init_visualizer()

    left_ee_task = pink.FrameTask(
        'left_wrist_yaw_link',
        position_cost=50.0,
        orientation_cost=10.0,
        lm_damping=0.1
    )

    posture_task = pink.PostureTask(
        cost=1e-3
    )

    configuration = pink.Configuration(
        robot_model.model,
        robot_model.data,
        robot_model.q,
    )

    tasks = [left_ee_task, posture_task]
    for task in tasks:
        task.set_target_from_configuration(configuration)
    left_ee_task.transform_target_to_world.translation = np.array([1, 0.2, 0])
    # select solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    dt = 0.01

    input('Press Enter to start the simulation...')

    while True:
        configuration = pink.Configuration(
            robot_model.model,
            robot_model.data,
            robot_model.q,
        )
        # robot_model.sync_subscriber()
        vel = pink.solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver
        )
        posture_task.set_target_from_configuration(configuration)
        robot_model._q = robot_model._q + vel * 1e-3
        robot_model.forward_kinematics()
        robot_model.update_viz()

        time.sleep(dt)
