import numpy as np

from mujoco._structs import MjModel, MjData
from adaptive_mpc import math_utils


def ref_dynamics(
    X_hat,
    F_hat,
    u_qp,
    u_adaptive,
    theta_hat,
    D,
    H_bar,
    B,
    G
):
    return D@X_hat + H_bar@F_hat + B@(u_qp + u_adaptive + G + theta_hat)

def state_vector(
    p_c,
    theta,
    p_c_dot,
    w_b,
):
    return np.hstack([
        p_c,
        theta,
        p_c_dot,
        w_b,
    ])

def get_D(roll, pitch, yaw):
    D = np.zeros((12, 12))
    D[0:3, 6:9] = np.eye(3)
    D[3:6, 9:12] = math_utils.R_z(yaw)
    return D

def get_H(M, A):
    H = np.block([
        [np.zeros((6, 12))],
        [np.linalg.inv(M) @ A],
    ])
    return H

def get_A(com_pos_w: np.ndarray, feet_pos_w: list[np.ndarray]):
    return np.block([
        [np.eye(3) for _ in range(len(feet_pos_w))],
        [math_utils.skew(feet_pos_w[i] - com_pos_w) for i in range(len(feet_pos_w))]
    ])

def get_B():
    return np.block([
        [np.zeros((6, 6))],
        [np.eye(6)],
    ])
    
def get_G():
    g = np.array([0, 0, -9.81]).reshape(3, 1)
    return np.block([
        [g],
        [np.zeros((3, 1))],
    ])

def get_M(m, I_G):
    return np.block([
        [m*np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), I_G],
    ])
