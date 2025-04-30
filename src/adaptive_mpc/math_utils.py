import numpy as np

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
    """
    Compute φ = vee(log(R)) ∈ ℝ³ for R∈SO(3):
    θ = arccos((trace(R) − 1)/2)
    φ = θ * (vee(R − R^T) / (2 sin θ))
    """
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

def euler_to_rotation_matrix(roll, pitch, yaw):
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    return np.array([
        np.arctan2(R[2,1], R[2,2]),
        np.arcsin(R[2,0]),
        np.arctan2(R[1,0], R[0,0]),
    ])

# def log_map(R: np.ndarray) -> np.ndarray:
#     cos_theta = (np.trace(R) - 1.0) / 2.0
#     theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#     if np.isclose(theta, 0.0):
#         return np.zeros(3, dtype=float)
#     axis = np.array([
#         R[2, 1] - R[1, 2],
#         R[0, 2] - R[2, 0],
#         R[1, 0] - R[0, 1],
#     ], dtype=float) / (2.0 * np.sin(theta))
#     return theta * axis

def get_R_z(roll: float, pitch: float, yaw: float):
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ]) 