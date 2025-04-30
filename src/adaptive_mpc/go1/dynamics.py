import numpy as np

from mujoco._structs import MjModel, MjData
from adaptive_mpc import math_utils
from adaptive_mpc.go1 import constants as const


def ref_dynamics(
    X_hat,
    F_hat,
    u_a,
    theta_hat,
    D,
    H_bar,
    B,
    G
):
    return D@X_hat + H_bar@F_hat + B@(u_a + G + theta_hat)

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
    D[3:6, 9:12] = math_utils.get_R_z(roll, pitch, yaw)
    return D

def get_H(M, A):
    H = np.zeros((12, 12))
    H[6:12, 0:6] = M.inv() @ A
    return H

def get_A(com_pos_w: np.ndarray, feet_pos_w: list[np.ndarray]):
    return np.block([
        [np.eye(3) for _ in range(const.N_FEET)],
        [math_utils.skew(feet_pos_w[i] - com_pos_w) for i in range(const.N_FEET)]
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

def get_M(model: MjModel, data: MjData):
    m = model.body("trunk").mass
    I_G = data.body("trunk").ximat.reshape(3, 3)
    return np.block([
        [m*np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), I_G],
    ])
