start_left = [0.1, 0.2, 0, 0.0, 0.0, 0.0]
start_right = [0.1, -0.2, 0, 0.0, 0.0, 0.0]
basic_pose_L = Lx, Ly, Lz, Lroll, Lpitch, Lyaw = 0.373, 0.323, 0.223, 0.0, 0.5899991834424116, 0.0
basic_pose_r = Rx, Ry, Rz, Rroll, Rpitch, Ryaw = 0.373, -0.323, 0.223, 0.0, 0.0, 0.5899991834424116

def in_safety_cylinder(x, y, z):
    radius = 0.2
    height = 1.0

    # Check radial distance from z-axis
    within_radius = (x**2 + y**2) <= radius**2
    # Check if z is within the height range
    within_height = -height <= z <= height

    return within_radius and within_height