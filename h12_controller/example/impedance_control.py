import time
import numpy as np

import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from controller import ArmController

def main():
    print('Initializing ArmController...')
    arm_controller = ArmController('assets/h1_2/h1_2.urdf',
                                   dt=0.01,
                                   vlim=1.0,
                                   visualize=False)

    for _ in range(300):
        arm_controller.goto_configuration(arm_controller.robot_model.zero_q)
        time.sleep(0.01)

    print('Entering impedance control mode...')
    # set gain for damp mode
    arm_controller.damp_mode(3.0)

    while True:
        start_time = time.time()
        arm_controller.impedance_step(np.array([0.23, 0.20, 0.09]))
        time.sleep(max(0, arm_controller.dt - (time.time() - start_time)))

if __name__ == '__main__':
    main()
