import numpy as np
import scipy.signal as signal
from collections import deque

class SecondOrderLPF:
    def __init__(self, fs: float, w_n: float, zeta: float) -> None:
        b = np.array([w_n**2])
        a = np.array([1, 2*zeta*w_n, w_n**2])
        b_d, a_d = signal.bilinear(b, a, fs)
        self.b0, self.b1, self.b2 = b_d
        _, self.a1, self.a2 = a_d
        self.z1 = None
        self.z2 = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.z1 is None:
            self.z1 = np.zeros_like(x)
        if self.z2 is None:
            self.z2 = np.zeros_like(x)
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y

def rk4(f, x, h):
    k1 = f(x)
    k2 = f(x + h/2 * k1)
    k3 = f(x + h/2 * k2)
    k4 = f(x + h * k3)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def convex_function(x, x_e, eps, s):
    # diff = x - x_e
    h = ( (1+eps)*(x-x_e).T @ np.diag(1./s.squeeze()) @ (x-x_e) - 1 ) / eps
    dh_dx = ( 2*(1+eps)*(x-x_e).T @ np.diag(1./s.squeeze()) ) / eps
    return h, dh_dx

def proj(x, x_d, h, dh_dx):
    """
    Projects x onto the constraint defined by h(x) = 0.
    """
    out_set = (h > 0) * (dh_dx*x_d > 0)
    
    # Compute the projection
    if np.all(out_set):
        x_proj = x_d - out_set * h * (dh_dx.T @ dh_dx) / (dh_dx @ dh_dx.T) @ x_d
    else:
        x_proj = x_d
    
    return x_proj

    # in_set = (h > 0) * (dh_dx*x_d > 0)
    
    # Compute the projection
    # return x_d - np.all(in_set) * h * (dh_dx.T @ dh_dx) / (dh_dx @ dh_dx.T) @ x_d

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

def skew_inv(v):
    """
    Recovers a vector from its skew-symmetric matrix.
    """
    return np.array([
        v[2, 1],
        v[0, 2],
        v[1, 0],
    ])
    

def log_so3(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    θ  = np.arccos((tr - 1.0) / 2.0)
    if np.isclose(θ, 0.0):
        return skew_inv(R - R.T) / 2.0
    return skew_inv(R - R.T) * θ / (2.0 * np.sin(θ))

def quat_to_rpy(q):
    return np.array([
        np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)),
        np.arcsin(2*(q[0]*q[2] - q[3]*q[1])),
        np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)),
    ])

def quat_to_rot_mat(q):
    # Normalize quaternion
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0):
        q /= norm

    w, x, y, z = q[0], q[1], q[2], q[3]

    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2),   2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])
    return R

def rpy_to_rot_mat(roll, pitch, yaw):
    return R_z(yaw) @ R_y(pitch) @ R_x(roll)

def rot_mat_to_euler(R: np.ndarray) -> np.ndarray:
    return np.array([
        np.arctan2(R[2,1], R[2,2]),
        np.arcsin(R[2,0]),
        np.arctan2(R[1,0], R[0,0]),
    ])

def R_z(yaw: float):
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ]) 

def R_y(pitch: float):
    return np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])

def R_x(roll: float):
    return np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
