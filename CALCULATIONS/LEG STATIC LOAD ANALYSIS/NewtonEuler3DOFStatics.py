# Kai De La Cruz
# 3DOF Leg Static Load Analysis using Newton-Euler Method
# Date: 10/10/2023
# Updated: 06/10/2024

import numpy as np
import itertools

# Define gravitational acceleration
g = np.array([0, 0, 0]) # m/s^2

# Define link lengths (in meters)
L1 = 0.083  # Length of link 1
L2 = 0.146  # Length of link 2
L3 = 0.165  # Length of link 3
L = np.array([L1, L2, L3])

# Define masses of each link (in kg)
m1 = 5.0  # Mass of link 1
m2 = 4.0  # Mass of link 2
m3 = 3.0  # Mass of link 3
m = np.array([m1, m2, m3])

# Define magnitude of distnace to center of mass for each link (as a fraction of link length)
r1 = L1 / 2
r2 = L2 / 2
r3 = L3 / 2
r = np.array([r1, r2, r3])

# Define joint angles (in radians)
theta1 = np.radians(0)  # Joint angle at base
theta2 = np.radians(45)  # Joint angle at knee
theta3 = np.radians(90)  # Joint angle at ankle
theta = np.array([theta1, theta2, theta3])

def calculate_loads(GRF,g,mass,length,r,theta):
    forces = np.zeros((3,3), dtype=float)
    torques = np.zeros((3,3), dtype=float)

    # Start with Link 3 (end effector)
    F23 = np.add(GRF, mass[2]*g)
    M23 = np.cross(r[2] * np.array([-np.cos(np.radians(180)-theta[2]), np.sin(np.radians(180)-theta[2]), 0]), F23) - np.cross(r[2] * np.array([np.cos(np.radians(180)-theta[2]), -np.sin(np.radians(180)-theta[2]), 0]), GRF)
    forces[2] = F23
    torques[2] = M23

    # Link 2
    F12 = np.add(F23, mass[1]*g)
    M12 = -M23 - np.cross(r[1] * np.array([np.cos(theta[1]), np.sin(theta[1]), 0]), F12) + np.cross(r[1] * np.array([-np.cos(theta[1]), -np.sin(theta[1]), 0]), -F23)
    forces[1] = F12
    torques[1] = M12

    # Link 1 (out of plane motor)
    F01 = np.add(F12, mass[0]*g)
    M01 = np.cross(r[0] * np.array([0, -np.sin(theta[0]), -np.cos(theta[0])]), F01) -np.cross(r[0] * np.array([0, np.sin(theta[0]), np.cos(theta[0])]), -F12)
    forces[0] = F01
    torques[0] = M01

    M = np.zeros((3,), dtype=float)
    M[0] = rss(torques[0])  # Only z-torque for out-of-plane motor
    M[1] = rss(torques[1])  # Only z-torque for planar motor
    M[2] = rss(torques[2])  # Only z-torque for planar motor

    return forces, M

def rss(v):
    return np.sqrt(np.sum(np.square(v)))

def metric_to_imperial(value, unit_type):
    if unit_type == 'length':
        return value * 3.28084  # meters to feet
    elif unit_type == 'force':
        return value * 0.224809  # Newtons to pounds
    elif unit_type == 'torque':
        return value * 0.737562  # Nm to ft-lbs
    else:
        raise ValueError("Unsupported unit type")

def max_loads(GRF, g, m, L, r):
    """
    Brute-force over angle grid (deg) and return per-joint max force magnitude
    and max torque magnitude, along with the angles (deg) at which they occur.
    """
    # Grids in degrees (range stop is exclusive)
    # theta1_deg = range(-90, 95, 5)   # [-90, 90]
    # theta2_deg = range(0, 365, 5)    # [0, 360]
    # theta3_deg = range(0, 140, 5)    # [0, 135]

    theta1_deg = range(-45, 50, 5)   # [-45, 45]
    theta2_deg = range(0, 95, 5)    # [0, 90]
    theta3_deg = range(0, 140, 5)    # [0, 135]

    max_F = np.zeros(3, dtype=float)
    max_F_at = [(None, None, None)] * 3

    max_M = np.zeros(3, dtype=float)
    max_M_at = [(None, None, None)] * 3

    for d1, d2, d3 in itertools.product(theta1_deg, theta2_deg, theta3_deg):
        theta = np.radians([d1, d2, d3])  # convert once here

        F, M = calculate_loads(GRF, g, m, L, r, theta)
        # F is (3x3) xyz forces per joint; M is (3,) z-torque per joint (planar)
        Fmags = [np.linalg.norm(F[i]) for i in range(3)]
        Mmags = [abs(M[i]) for i in range(3)]

        for i in range(3):
            if Fmags[i] > max_F[i]:
                max_F[i] = Fmags[i]
                max_F_at[i] = (d1, d2, d3)

            if Mmags[i] > max_M[i]:
                max_M[i] = Mmags[i]
                max_M_at[i] = (d1, d2, d3)

    return {
        "max_force": max_F,
        "max_force_at_deg": max_F_at,
        "max_torque": max_M,
        "max_torque_at_deg": max_M_at,
    }

if __name__ == "__main__":
    GRF = np.array([0, 174, 0])  # Example ground reaction force in Newtons
    forces, torques = calculate_loads(GRF, g, m, L, r, theta)
    force_imperial = [metric_to_imperial(rss(f), 'force') for f in forces]
    torque_imperial = [metric_to_imperial(rss(t), 'torque') for t in torques]
    print()
    print(f"Ground Reaction Force: {rss(GRF):.2f} N ({metric_to_imperial(rss(GRF), 'force'):.2f} lbs)")
    print()
    # print(f"Joint Angles (degrees): {np.degrees(theta)}")

    # F1 = rss(forces[0])
    # F2 = rss(forces[1])
    # F3 = rss(forces[2])

    # print(f"Force at Joint 1: {F1:.2f} N, Torque at Joint 1: {torques[0]:.2f} Nm")
    # print(f"Force at Joint 2: {F2:.2f} N, Torque at Joint 2: {torques[1]:.2f} Nm")
    # print(f"Force at Joint 3: {F3:.2f} N, Torque at Joint 3: {torques[2]:.2f} Nm")
    # print()

    # F1 = rss(force_imperial[0])
    # F2 = rss(force_imperial[1])
    # F3 = rss(force_imperial[2])
    
    # print(f"Force at Joint 1: {F1:.2f} lbs, Torque at Joint 1: {torque_imperial[0]:.2f} ft-lbs")
    # print(f"Force at Joint 2: {F2:.2f} lbs, Torque at Joint 2: {torque_imperial[1]:.2f} ft-lbs")
    # print(f"Force at Joint 3: {F3:.2f} lbs, Torque at Joint 3: {torque_imperial[2]:.2f} ft-lbs")

    res = max_loads(GRF, g, m, L, r)

    forces_imperial = [metric_to_imperial(f, 'force') for f in res["max_force"]]
    torques_imperial = [metric_to_imperial(t, 'torque') for t in res["max_torque"]]

    # [Joint 1, Joint 2, Joint 3]
    print("Max joint forces (N):", [f"{f:.2f}" for f in res["max_force"]])
    print("Max joint forces (lbs):", [f"{f:.2f}" for f in forces_imperial])
    print("At angles (deg):", res["max_force_at_deg"])
    
    print()

    # [Joint 1, Joint 2, Joint 3]
    print("Max joint torques (N·m):", [f"{t:.2f}" for t in res["max_torque"]])
    print("Max joint torques (ft-lbs):", [f"{t:.2f}" for t in torques_imperial])
    print("At angles (deg):", res["max_torque_at_deg"])
    
    print()
    

