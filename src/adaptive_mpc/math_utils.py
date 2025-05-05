import numpy as np
import scipy.signal as signal

class SecondOrderLPF:
    def __init__(self, w_n, zeta):
        self.b = [w_n**2]
        self.a = [1, 2*zeta*w_n, w_n**2]
        self.lpf = signal.TransferFunction(self.b, self.a)
        
    def __call__(self, x):
        return signal.lfilter(self.b, self.a, x)

def rk4(f, x, h):
    k1 = f(x)
    k2 = f(x + h/2 * k1)
    k3 = f(x + h/2 * k2)
    k4 = f(x + h * k3)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def proj(v, u):
    return (v @ u) / (u @ u) * u

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
