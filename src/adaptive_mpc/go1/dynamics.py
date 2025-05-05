import numpy as np

from scipy.integrate import solve_ivp
from mujoco._structs import MjModel, MjData
from adaptive_mpc import math_utils

class Go1ReferenceModel:
    def __init__(self, dt: float):
        self.dt = dt

    def ref_dynamics(
        self,
        t,
        X_hat,
        F_hat,
        u_qp,
        theta_hat_lpf,
        theta_hat,
        D,
        H_bar,
        B,
        G
    ):
        x_dot = D@X_hat + H_bar@F_hat + B@(u_qp + theta_hat - theta_hat_lpf + G).squeeze()
        return x_dot
    
    def __call__(self, 
                 p_c, theta, p_c_dot, w_b, # Initial state
                 F_hat, u_qp, u_adaptive, theta_hat, # Control inputs
                 D, H_bar, B, G, # Dynamics
                 ):
        X_hat = state_vector(p_c, theta, p_c_dot, w_b)
        sol = solve_ivp(self.ref_dynamics, [0, self.dt], X_hat, args=(F_hat, u_qp, u_adaptive, theta_hat, D, H_bar, B, G), method='RK45', t_eval=[self.dt])
        y = sol.y[:, -1]
        return (
            y[0:3], # p_c
            y[3:6], # theta
            y[6:9], # p_c_dot
            y[9:12], # w_b    
        )

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
