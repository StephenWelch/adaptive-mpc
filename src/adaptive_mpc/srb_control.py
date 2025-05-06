import casadi as ca
import numpy as np
import rerun as rr
from adaptive_mpc import math_utils
from scipy.linalg import solve_continuous_lyapunov

class SingleRigidBodyQP:
    def __init__(self, n_contact_pts: int):
        self.N = 12
        self.n_contact_pts = n_contact_pts
        self.sol = None

        PENALTY_L2_GRF = 0.0
        PENALTY_L2_GRF_SLEW = 0.0

        self.opti = ca.Opti("conic")
        self.opti.solver("osqp", dict(warm_start_primal=True, warm_start_dual=True), dict(verbose=False))

        # F_MIN = np.array([-500.0, -500.0, -500.0]).repeat(n_contact_pts)
        # F_MAX = np.array([500.0, 500.0, 0.0]).repeat(n_contact_pts)
        F_MIN = np.array([-50.0, -50.0, -50.0]).repeat(n_contact_pts)
        F_MAX = np.array([50.0, 50.0, 50.0]).repeat(n_contact_pts)

        self.A = self.opti.parameter(6, self.N) # Contact Jacobian
        self.b = self.opti.parameter(6, 1) # Desired wrench
        self.C = self.opti.parameter(n_contact_pts*3) # Contact flag
        self.S = self.opti.parameter(6, 6) # Weight matrix
        self.F_last = self.opti.parameter(n_contact_pts*3) # Previous GRFs

        self.F = self.opti.variable(n_contact_pts*3) # Decision variable - GRFs

        self.opti.minimize(
            (self.A@self.F - self.b).T @ self.S @ (self.A@self.F - self.b) + 
            PENALTY_L2_GRF * ca.sumsqr(self.F) + 
            PENALTY_L2_GRF_SLEW * ca.sumsqr(self.F - self.F_last)
        )
        self.opti.subject_to(self.C * self.F >= F_MIN)
        self.opti.subject_to(self.C * self.F <= F_MAX)
        
        S_ = np.diag([
            1, 1, 1,
            10, 10, 10
        ])
        self.opti.set_value(self.S, S_)

    def __call__(self, 
        p_c, theta, p_c_dot, w_b, # SRB state
        M, A,                     # SRB dynamics
        b,                        # Desired wrench
        contact_flags,            # Contact flags
    ):
        """Compute contact point GRFs to achieve a desired wrench around a single rigid body.

        Args:
            p_c (np.ndarray): Center of mass position in world frame.
            theta (np.ndarray): Euler angles in world frame.
            p_c_dot (np.ndarray): Center of mass velocity in world frame.
            w_b (np.ndarray): Angular velocity in world frame.
            F_des (np.ndarray): Desired wrench in world frame.
            M (np.ndarray): Inertia matrix.
            A (np.ndarray): Jacobian of contact points.
        """
        C = np.concatenate([np.ones(3)*f for f in contact_flags])
        if self.sol is not None:
            F_last = self.sol.value(self.F)
        else:
            F_last = np.zeros(3*self.n_contact_pts)

        self.opti.set_value(self.A, A)
        self.opti.set_value(self.b, b)
        self.opti.set_value(self.C, C)
        self.opti.set_value(self.F_last, F_last)
        self.opti.set_initial(self.F, F_last)
        self.sol = self.opti.solve()

        return self.sol.value(self.F)
        
class SingleRigidBodyL1Adaptation:
    def __init__(self, dt: float, B: np.ndarray, K: np.ndarray):
        self.dt = dt
        self.B = B
        self.A_m = np.block([
            [np.zeros((6, 6)), np.eye(6)],
            [K]
        ])
        print(f"{self.A_m=}")
        self.Q_L = np.eye(12)*1.0
        self.P = solve_continuous_lyapunov(self.A_m.T, self.Q_L)
        # print(f"{self.P=}")
        # self.Gamma = np.diag([1.0, 1.0, 5.0, 2.0, 5.0, 1.0]) * 1e3
        self.Gamma = np.diag([1.0, 1.0, 5.0, 2.0, 5.0, 1.0]) * 1e4
        self.alpha_hat = np.zeros((6, 1))
        self.beta_hat = np.zeros((6, 1))
        
        self.theta_lpf = math_utils.SecondOrderLPF(fs=1.0/dt, w_n=60.0, zeta=0.7)
        
    def _alpha_hat_dot(self, alpha_hat, e, e_hat):
        # e_hat is ref model err
        e_tilde = e_hat - e
        y_alpha = -self.B.T @ self.P @ e_tilde * np.linalg.norm(e)

        x_e = np.zeros_like(alpha_hat)
        eps = 1.0
        s = np.full_like(x_e, 10) 
        h, dh_dx = math_utils.convex_function(alpha_hat, x_e, eps, s)
        return self.Gamma @ math_utils.proj(alpha_hat, y_alpha, h, dh_dx)
        # return self.Gamma @ y_alpha
        
    def _beta_hat_dot(self, beta_hat, e, e_hat):
        e_tilde = e_hat - e
        y_beta = -self.B.T @ self.P @ e_tilde

        x_e = np.zeros_like(beta_hat)
        eps = 100.0
        s = np.full_like(x_e, 10000) 
        h, dh_dx = math_utils.convex_function(beta_hat, x_e, eps, s)
        return self.Gamma @ math_utils.proj(beta_hat, y_beta, h, dh_dx)
        # return self.Gamma @ y_beta
        
    def __call__(self, e, e_hat):
        #TODO integrate alpha_hat, beta_hat with scipy
        self.alpha_hat = math_utils.rk4(
            lambda x: self._alpha_hat_dot(x, e, e_hat),
            self.alpha_hat,
            self.dt
        )
        
        self.beta_hat = math_utils.rk4(
            lambda x: self._beta_hat_dot(x, e, e_hat),
            self.beta_hat,
            self.dt
        )

        theta_hat = self.alpha_hat * np.linalg.norm(e) + self.beta_hat
        theta_hat_lpf = self.theta_lpf(theta_hat)
        # print(f"{self._alpha_hat_dot(self.alpha_hat, e, e_hat)=}")
        # print(f"{self._beta_hat_dot(self.beta_hat, e, e_hat)=}")
        # print(f"{self.alpha_hat=}")
        # print(f"{self.beta_hat=}")
        print(f"{theta_hat=}")
        print(f"{theta_hat_lpf=}")
        # print(f"{np.linalg.norm(e)=}")

        return theta_hat, theta_hat_lpf

def compute_error(
    p_c, theta, p_c_dot, w_b, # SRB state
    des_p_c, des_theta, des_p_c_dot, des_w_b, # Desired SRB state
):
    R_des = math_utils.rpy_to_rot_mat(*des_theta)
    b_R_w = math_utils.rpy_to_rot_mat(*theta)
    # b_R_w = math_utils.R_z(theta[2])
    return np.hstack([
        p_c - des_p_c,
        math_utils.log_so3(R_des.T @ b_R_w).flatten(),
        p_c_dot - des_p_c_dot,
        w_b - des_w_b,
    ]).reshape(12, 1)
    
