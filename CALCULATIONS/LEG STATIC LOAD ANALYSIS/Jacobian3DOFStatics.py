# Kai De La Cruz
# 3DOF Leg Static Load Analysis using Jacobian Method
# Date: 9/26/2025
# Updated: 9/26/2025

import numpy as np
import sympy as sp

def dh_transform_sym(theta, d, a, alpha):
    """DH transformation using SymPy (symbolic)."""
    ct = sp.cos(theta)
    st = sp.sin(theta)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)

    T = sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,     sa,     ca,    d],
        [0,     0,      0,    1]
    ])
    return sp.simplify(T)

def frame_tf(Tn,Tm):
    """
    Compute the transformation from frame n to frame m.
    Tn: Homogeneous transformation matrix of frame n
    Tm: Homogeneous transformation matrix of frame m
    Returns the transformation matrix from frame m to frame n.
    """
    Tnm = Tm @ Tn
    return Tnm

def output_z_p(T):
    """
    Extract the z-axis and position vector from a homogeneous transformation matrix.
    T: Homogeneous transformation matrix
    Returns the z-axis (3x1) and position vector (3x1).
    """
    z = T[0:3, 2]
    p = T[0:3, 3]
    return z, p

def jacobian_sym(z_list, p_list, p_e):
    # z_list, p_list: lists of 3×1 sympy Matrices for joints 1..n
    Jp = sp.Matrix.hstack(*[z_list[i].cross(p_e - p_list[i]) for i in range(len(z_list))])
    Jo = sp.Matrix.hstack(*[z_list[i] for i in range(len(z_list))])
    return sp.Matrix.vstack(Jp, Jo)

def torque_from_force(J, F):
    """
    Compute the joint torques from an end-effector force using the Jacobian transpose method.
    J: Jacobian matrix (6xn)
    F: End-effector force vector (6x1)
    Returns the joint torque vector (nx1).
    """
    tau = J.T @ F
    return tau

if __name__ == "__main__":
    theta1 = sp.Symbol('theta1')
    theta2 = sp.Symbol('theta2')
    theta3 = sp.Symbol('theta3')
    L1 = sp.Symbol('L1')
    L2 = sp.Symbol('L2')
    L3 = sp.Symbol('L3')
    L4 = sp.Symbol('L4')

    T01 = dh_transform_sym(np.pi/2,0,0,np.pi/2)
    T12 = dh_transform_sym(theta1 - sp.pi/2, L1, 0, -sp.pi/2)
    T23 = dh_transform_sym(theta2 - sp.pi/2, L2, L3, 0)
    T3e = dh_transform_sym(theta3, 0, L4, 0)
    # sp.pprint(T01)
    # sp.pprint(T12)
    # sp.pprint(T23)
    # sp.pprint(T3e)

    T = [T01, T12, T23, T3e]
    T02 = frame_tf(T[0], T[1])
    T03 = frame_tf(T02, T[2])
    T0e = frame_tf(T03, T[3])

    # sp.pprint(T03)

    z1, p1 = output_z_p(T01)
    z2, p2 = output_z_p(T02)
    z3, p3 = output_z_p(T03)
    ze, pe = output_z_p(T0e)  # pe used, ze not needed in J

    # print("z1:", z1)
    # print("p1:", p1)
    # print("z2:", z2)
    # print("p2:", p2)
    # print("z3:", z3)
    # print("p3:", p3)
    # print("ze:", ze)
    # print("pe:", pe)

    J = sp.simplify(jacobian_sym([z1, z2, z3], [p1, p2, p3], pe))
    # sp.pprint(J)

    F = sp.Matrix([0, 0, 88, 0, 0, 0])  # 88 N force in z-direction in universal coordinate frame
    # sp.pprint(F)

    print("Symbolic Joint Torques (Nm):")
    tau = sp.simplify(torque_from_force(J, F))
    # sp.pprint(tau)
    # print()

    # Plug in thetas and lengths
    subs = {
        theta1: np.deg2rad(0),
        theta2: np.deg2rad(45),
        theta3: np.deg2rad(90),
        L1: 0.061,    # m1 to m2 rotation axis
        L2: 0.083,    # link 1 length
        L3: 0.146,    # link 2 length
        L4: 0.165     # link 3 length
    }
    tau_num = tau.evalf(subs=subs)
    print("Numerical Joint Torques (Nm):")
    sp.pprint(tau_num)
    print()

    print("Joint Torques (ft-lbs):")
    tau_ftlb = tau_num * 0.73756214927727
    sp.pprint(tau_ftlb)