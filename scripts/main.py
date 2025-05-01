import mujoco
import mujoco.viewer
import casadi as ca
import time
import scipy
import rerun as rr
import numpy as np
from mujoco._structs import MjModel, MjData

from adaptive_mpc.go1 import Go1Constants
from adaptive_mpc.go1 import dynamics
from adaptive_mpc import math_utils
from adaptive_mpc import sim_utils
from adaptive_mpc.utils import PeriodicCallback, Timer

np.set_printoptions(precision=3, linewidth=1000, floatmode="fixed")

def main():
    model = mujoco.MjModel.from_xml_path("assets/unitree_go1/scene.xml")
    data = mujoco.MjData(model)

    const = Go1Constants(model)
    N = 12
    M = 12

    opti = ca.Opti("conic")

    PENALTY_L2_GRF = 0.01
    PENALTY_L2_GRF_SLEW = 0.001
    F_MIN = -200.0
    F_MAX = 200.0
    A = opti.parameter(6, N) # Actuation force
    b = opti.parameter(6, 1) # External force
    C = opti.parameter(const.N_FEET*3) # Contact flag
    S = opti.parameter(6, 6) # Weight matrix
    F_last = opti.parameter(const.N_FEET*3) # Previous GRFs
    F = opti.variable(const.N_FEET*3) # Decision variable - GRFs

    opti.minimize((A@F - b).T @ S @ (A@F - b) + PENALTY_L2_GRF * ca.sumsqr(F) + PENALTY_L2_GRF_SLEW * ca.sumsqr(F - F_last))
    opti.subject_to(C * F >= F_MIN)
    opti.subject_to(C * F <= F_MAX)

    m = model.body("trunk").mass

    G = dynamics.get_G()

    K_P = np.diag([
        500, 500, 500,
        -150, 150, 150
    ])*-1
    K_D = np.diag([
        1, 1, 1,
        -1, -1, -1,
        # 0, 0, 0
    ])*-1
    K = np.block([-K_P, -K_D])

    S_ = np.diag([
        1, 1, 1,
        10, 10, 10
    ])
    opti.set_value(S, ca.DM(S_.tolist()))

    def update_opt(model: MjModel, data: MjData, sol_last: ca.OptiSol | None = None, log:bool=False):
        I_G = data.body("trunk").ximat.reshape(3, 3)
        R = data.body("trunk").xmat.reshape(3, 3)
        # theta = math_utils.quat_to_rpy(data.qpos[3:7])
        p_c = data.body("trunk").xipos
        p_i = [data.site(const.FEET_SITE_NAMES[n]).xpos for n in const.LEG_NAMES]
        p_c_dot = data.qvel[7:10]
        w_b = R.T @ data.qvel[10:13]

        M = dynamics.get_M(m, I_G)
        A_ = dynamics.get_A(p_c, p_i)
        
        p_c_des = np.array([0.022, 0, 0.278])
        theta_des = np.zeros(3)
        R_des = np.array(math_utils.euler_to_rotation_matrix(*theta_des))
        p_c_dot_des = np.array([0, 0, 0])
        w_b_des = np.array([0, 0, 0])
        e = np.hstack([
            p_c - p_c_des,
            math_utils.log_so3(R_des @ R.T).flatten(),
            p_c_dot - p_c_dot_des,
            w_b - w_b_des,
        ]).reshape(12, 1)
        u = K @ e
        b_ = M @ (u + G)

        

        if log:
            for i, name in enumerate(["u/lin/x", "u/lin/y", "u/lin/z", "u/ang/x", "u/ang/y", "u/ang/z"]):
                rr.log(name, rr.Scalar(u[i]))
            for i, name in enumerate([
                "error/pos/lin/x", "error/pos/lin/y", "error/pos/lin/z",
                "error/pos/ang/x", "error/pos/ang/y", "error/pos/ang/z",
                "error/vel/lin/x", "error/vel/lin/y", "error/vel/lin/z",
                "error/vel/ang/x", "error/vel/ang/y", "error/vel/ang/z",
            ]):
                rr.log(name, rr.Scalar(-e[i]))

        contact_flags = [data.sensor(const.FEET_CONTACT_SENSOR_NAMES[n]).data > 0.0 for n in const.LEG_NAMES]
        C_ = np.concatenate([np.ones(3)*f for f in contact_flags])
        if sol_last is not None:
            F_last_ = sol_last.value(F_last)
        else:
            F_last_ = np.zeros((const.N_FEET*3))

        # opti.set_value(A, ca.DM(A_.tolist()))
        # opti.set_value(b, ca.DM(b_.tolist()))
        # opti.set_value(C, ca.DM(C_.tolist()))
        # opti.set_value(F_last, ca.DM(F_last_.tolist()))
        opti.set_value(A, A_)
        opti.set_value(b, b_)
        opti.set_value(C, C_)
        opti.set_value(F_last, F_last_)

    def ctrl_loop(model: MjModel, data: MjData, F_des: np.ndarray):
        for foot_idx, leg_name in enumerate(const.LEG_NAMES):
            site_name = const.FEET_SITE_NAMES[leg_name]
            act_idxs = const.LEG_ACT_IDXS[leg_name]
            jnt_idxs = const.LEG_JNT_IDXS[leg_name]

            J = sim_utils.compute_site_jacobian(model, data, model.site(site_name).id)[:3, jnt_idxs]
            F = F_des[foot_idx*3:foot_idx*3+3]

            data.ctrl[act_idxs] = J.T @ F

    rr.init("adaptive_mpc", spawn=True)
    rr.set_time_seconds("mj_time", 0)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"{model.opt.timestep=}")

        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)

        sol = None
        F_des = np.zeros(const.N_FEET*3)
        # opti.solver("ipopt", {
        #     'ipopt.print_level': 0, 
        #     'print_time': 0, 
        #     'ipopt.sb': 'yes'
        # })
        # opti.solver("qpoases", dict(printLevel="none"))
        opti.solver("osqp", dict(), dict(verbose=False))

        solve_timer = Timer(
            PeriodicCallback(
                lambda duration: print(f"Solve time: {duration}"),
                100
            ),
        )

        ctr = 0
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            rr.set_time_seconds("mj_time", data.time)

            if ctr % 1 == 0:
                with solve_timer:
                    update_opt(model, data, sol)
                    if F_des is not None:
                        opti.set_initial(F, F_des)
                    sol = opti.solve()
                    F_des = sol.value(F)
            ctr += 1

            ctrl_loop(model, data, F_des)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            viewer.sync()

            time_elapsed = time.time() - step_start
            rr.log("rtf", rr.Scalar(time_elapsed/model.opt.timestep))
            # if time_elapsed > model.opt.timestep:
            #     print(f"Overrun! RTF: {time_elapsed/model.opt.timestep}") 
            # time_until_next_step = model.opt.timestep - time_elapsed
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()