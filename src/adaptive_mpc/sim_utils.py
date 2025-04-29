import mujoco
import numpy as np

def compute_site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    """Compute the Jacobian matrix for a given site in the MuJoCo model.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        site_id: ID of the site to compute Jacobian for
        
    Returns:
        np.ndarray: Jacobian matrix of shape (6, nv) where nv is the number of degrees of freedom
    """
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return np.vstack((jacp, jacr))  # Shape: (6, nv) 