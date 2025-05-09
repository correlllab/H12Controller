import os
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

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
        self._dq = np.zeros(self.model.nv)
        self._tau = np.zeros(self.model.nv)

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
    def tau(self):
        return np.copy(self._tau)

    @property
    def zero_q(self):
        return np.zeros(self.model.nq)

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

    def update_kinematics(self):
        # udpate data with the current joint positions
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def update_visualizer(self):
        self.viz.display()

    def get_frame_transformation(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        transformation = self.data.oMf[frame_id]
        return transformation.np

    def get_frame_position(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        transformation = self.data.oMf[frame_id]
        return transformation.translation

    def get_frame_rotation(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        transformation = self.data.oMf[frame_id]
        return transformation.rotation

    def get_frame_jacobian(self, frame_name: str):
        '''
        Get the frame jacobian in the world frame
        '''
        frame_id = self.model.getFrameId(frame_name)
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return jacobian

    def get_joint_jacobian(self, joint_name: str):
        '''
        Get the joint jacobian in the local frame of the joint
        '''
        joint_id = self.model.getJointId(joint_name)
        jacobian = pin.computeJointJacobian(
            self.model,
            self.data,
            self.q,
            joint_id
        )
        return jacobian

if __name__ == '__main__':
    # a simple shadowing program
    ChannelFactoryInitialize(id=0)
    # Example usage
    robot_model = RobotModel('assets/h1_2/h1_2.urdf')
    # robot_model = RobotModel('assets/h1_2/h1_2.xml')
    robot_model.init_visualizer()
    robot_model.init_subscriber()

    while True:
        robot_model.sync_subscriber()
        robot_model.update_kinematics()
        robot_model.update_visualizer()
        time.sleep(0.01)
