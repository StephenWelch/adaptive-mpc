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

def rotation_error_rpy(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    R_err = R2 @ R1.T
    roll = np.arctan2(R_err[2,1], R_err[2,2])
    pitch = np.arctan2(-R_err[2,0], np.hypot(R_err[2,1], R_err[2,2]))
    yaw = np.arctan2(R_err[1,0], R_err[0,0])
    return np.array([roll, pitch, yaw])

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
