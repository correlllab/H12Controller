# H12ControllerWrapper

This repo implements a controller for the H1-2 robot.

You can control the position and orientation of the two end-effectors.

## Installation

- Install Python dependencies from `environment.yml`:
    ```bash
    conda env create -f environment.yml
    ```
- All dependencies are available on pip, so using pip for installation should also work.
- Install the Unitree Python SDK from [here](https://github.com/unitreerobotics/unitree_sdk2_python/tree/master)
  so the code can communicate with the robot.

## Files

- `robot_model.py` tracks robot states and provides useful functions for kinematics, Jacobians, etc., using Pinocchio.
- `controller.py` solves inverse kinematics and provides functions to control end-effectors and query their states.
- `channel_interface.py` implements a publisher for motor commands and a subscriber for motor states using the Unitree Python SDK.
  In the future, this part should be implemented using the Unitree ROS2 SDK for ROS2 integration.
- `archive/` contains the old controller implementation.
- `assets/` contains robot description files.
- `utility/` contains useful scripts to inspect robot descriptions, process collision pairs, and lock the robot configuration.
- `data/` contains data such as joint configurations.

## TODO

- Kinematics Tracker & Solver (`robot_model.py`)
    - [x] Initialize using robot description file.
    - [x] Subscribe to robot motor states; track joint positions $q$ and torque $\tau$.
    - [x] Given a body name, return the transformation matrix.
    - [x] Given a body name and target position, solve inverse kinematics (avoid self-collision using [pink](https://github.com/stephane-caron/pink)).
    - [x] Given a body name, return the Jacobian.
    - [x] Given a body name, return the estimated wrench.
    - [ ] Given a configuration, check validity (joint limits, collisions, etc.).

- Joint Controller (`controller.py`)
    - [x] Publish motor commands.
    - [ ] Control modes: position, velocity, force ([Motor SDK](https://support.unitree.com/home/en/Motor_SDK_Dev_Guide/overview)).
    - [x] Import the kinematics tracker & solver to track robot states.
    - [ ] Implement position, velocity, and force control routines for left and right arms.
    - [x] Add safety threshold on moving velocity.
