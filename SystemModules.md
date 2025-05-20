# System Modules

## Kinematics Tracker & Solver

- [x] Initialize on robot description file.
- [x] Subscribe to robot motor states; track joint positions $q$ and torque $tau$.
- [x] Given body name, return transformation matrix.
- [x] Given body name and target position, solve inverse kinematics. (avoid self collision using [pink](https://github.com/stephane-caron/pink))
- [x] Given body name, return Jacobian.
- [x] Given body name, return estimated wrench.
- [ ] Given a configuration, query validity (joint limit, collision, etc.).

## Joint Controller

- [x] Publish motor commands.
- [ ] Mode of control: position, velocity, force. ([Motor SDK](https://support.unitree.com/home/en/Motor_SDK_Dev_Guide/overview))
- [x] Import kinematics tracker & solver to track robot states.
- [ ] Position, velocity, force control routines for left and right arms.
- [x] Safety threshold on moving velocity.
