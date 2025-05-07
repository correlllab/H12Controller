import os
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from unitree_h1_2_interface import StateSubscriber

class KinodynamicModel:
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
        self.data = self.model.createData()

    def init_visualizer(self):
        try:
            self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model,
                                         copy_models=False, data=self.data)
            print(self.data)
            print(self.viz.data)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel('unitree_h1_2')
        except ImportError as err:
            print('importError: MeshcatVisualizer requires the meshcat package.')
            print(err)
            exit(0)

    def init_subscriber(self):
        self.state_subscriber = StateSubscriber()

    def sync_subscriber(self):
        q = np.zeros(self.model.nq)
        q[0:20] = self.state_subscriber.q[0:20]
        q[32:39] = self.state_subscriber.q[20:27]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def update_viz(self):
        self.viz.display()


if __name__ == '__main__':
    # Example usage
    kinodynamic_model = KinodynamicModel('assets/h1_2/h1_2.urdf')
    # kinodynamic_model = KinodynamicModel('assets/h1_2/h1_2.xml')
    kinodynamic_model.init_visualizer()
    kinodynamic_model.init_subscriber()

    while True:
        kinodynamic_model.sync_subscriber()
        kinodynamic_model.update_viz()

