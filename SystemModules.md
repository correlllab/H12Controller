# System Modules

## Kinematics Tracker & Solver

- Initialize on robot description file.
- Subscribe to robot motor states; track joint positions $q$ and torque $tau$.
- Given body name, return transformation matrix.
- Given body name and target position, solve inverse kinematics. (avoid self collision using [pink](https://github.com/stephane-caron/pink))
- Given body name, return Jacobian.
- Given body name, return estimated wrench.
- Given a configuration, query validity (joint limit, collision, etc.).

## Joint Controller

- Publish motor commands.
- Mode of control: position, velocity, force. ([Motor SDK](https://support.unitree.com/home/en/Motor_SDK_Dev_Guide/overview))
- Import kinematics tracker & solver to track robot states.
- Position, velocity, force control routines for left and right arms.
- Safety threshold on moving velocity.
