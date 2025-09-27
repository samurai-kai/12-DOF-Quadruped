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

def dh_transform(theta, d, a, alpha, decimals=12):
    """DH transformation for one joint, with rounding to remove floating errors."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,     sa,     ca,    d],
        [0,     0,      0,    1]
    ])

    return np.round(T, decimals)

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
    z = T[0:2, 2]
    p = T[0:2, 3]
    return z, p

def jacobian(z,p,pe):
  ''' Compute the Jacobian matrix for 3 dof leg'''
  Jp = np.zeros((3,4), dtype=float)
  Jo = np.zeros((3,4), dtype=float)

  for i in range(4):
      Jp[:,i] = np.cross(z[:,i], (pe - p[:,i]))
      Jo[:,i] = z[:,i]

  J = np.vstack((Jp, Jo))
  return J

if __name__ == "__main__":
  T01 = dh_transform(np.pi/2,0,0,np.pi/2)
  
  print("T01:\n", T01)

  theta1 = sp.Symbol('theta1')
  L1 = sp.Symbol('L1')

  T12 = dh_transform_sym(theta1 - sp.pi/2, L1, 0, -sp.pi/2)
  sp.pprint(T12)
