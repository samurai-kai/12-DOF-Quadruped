#Kai De La Cruz
# 3DOF Leg Static Load Analysis using Newton-Euler Method
# Date: 10/10/2023
# Updated: 06/10/2024

import numpy as np

# Define gravitational acceleration
g = np.array([0, -9.81, 0]) # m/s^2

# Define link lengths (in meters)
L1 = 0.5  # Length of link 1
L2 = 0.4  # Length of link 2
L3 = 0.3  # Length of link 3
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
theta1 = np.radians(30)  # Joint angle at base
theta2 = np.radians(45)  # Joint angle at knee
theta3 = np.radians(60)  # Joint angle at ankle
theta = np.array([theta1, theta2, theta3])

def calculate_loads(GRF,g,mass,length,r,theta):
  forces = np.zeros((3,3), dtype=float)
  torques = np.zeros((3,3), dtype=float)

  # Start with Link 3 (end effector)
  F23 = np.add(-GRF, mass[2]*g)
  M23 = np.cross(r[2] * np.array([-np.cos(theta[2]), np.sin(theta[2]), 0]), F23) + np.cross(r[2] * np.array([np.cos(theta[2]), -np.sin(theta[2]), 0]), GRF)
  forces[2] = F23
  torques[2] = M23

  # Link 2
  F12 = np.add(F23, mass[1]*g)
  M12 = M23 - np.cross(r[1] * np.array([-np.cos(theta[1]), np.sin(theta[1]), 0]), F12) - np.cross(r[1] * np.array([np.cos(theta[1]), -np.sin(theta[1]), 0]), -F23)
  forces[1] = F12
  torques[1] = M12

  # Link 1 (out of plane motor)
  F01 = np.add(F12, mass[0]*g)
  M01 = -np.cross(r[0] * np.array([0, np.sin(theta[0]), -np.cos(theta[0])]), F01) -np.cross(r[0] * np.array([0, -np.sin(theta[0]), np.cos(theta[0])]), -F12)
  forces[0] = F01
  torques[0] = M01

  return forces, torques

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



if __name__ == "__main__":
    GRF = np.array([0, 50, 0])  # Example ground reaction force in Newtons
    forces, torques = calculate_loads(GRF, g, m, L, r, theta)
    force_imperial = [metric_to_imperial(rss(f), 'force') for f in forces]
    torque_imperial = [metric_to_imperial(rss(t), 'torque') for t in torques]

    print(f"Ground Reaction Force: {rss(GRF):.2f} N ({metric_to_imperial(rss(GRF), 'force'):.2f} lbs)")

    F1 = rss(forces[0])
    F2 = rss(forces[1])
    F3 = rss(forces[2])
    M1 = rss(torques[0])
    M2 = rss(torques[1])
    M3 = rss(torques[2])
    print(f"Force at Joint 1: {F1:.2f} N, Torque at Joint 1: {M1:.2f} Nm")
    print(f"Force at Joint 2: {F2:.2f} N, Torque at Joint 2: {M2:.2f} Nm")
    print(f"Force at Joint 3: {F3:.2f} N, Torque at Joint 3: {M3:.2f} Nm")
    print()

    F1 = rss(force_imperial[0])
    F2 = rss(force_imperial[1])
    F3 = rss(force_imperial[2])
    M1 = rss(torque_imperial[0])
    M2 = rss(torque_imperial[1])
    M3 = rss(torque_imperial[2])
    print(f"Force at Joint 1: {F1:.2f} lbs, Torque at Joint 1: {M1:.2f} ft-lbs")
    print(f"Force at Joint 2: {F2:.2f} lbs, Torque at Joint 2: {M2:.2f} ft-lbs")
    print(f"Force at Joint 3: {F3:.2f} lbs, Torque at Joint 3: {M3:.2f} ft-lbs")
