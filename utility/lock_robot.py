import time

import os
import sys
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], '..')))
from controller import ArmController

def main():
    print('Initializing ArmController...')
    arm_controller = ArmController('assets/h1_2/h1_2.urdf',
                                   dt=0.01,
                                   vlim=1.0,
                                   visualize=True)

    print('Lock robot in current configuration')
    while True:
        arm_controller.lock_configuration()
        time.sleep(arm_controller.dt)

if __name__ == "__main__":
    main()
