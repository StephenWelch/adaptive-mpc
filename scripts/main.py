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

    # PENALTY_L2_GRF = 0.01
    # PENALTY_L2_GRF_SLEW = 0.001
    PENALTY_L2_GRF = 0.0
    PENALTY_L2_GRF_SLEW = 0.0
    F_MIN = np.array([-500.0, -500.0, -500.0]).repeat(const.N_FEET)
    F_MAX = np.array([500.0, 500.0, 0.0]).repeat(const.N_FEET)
    A = opti.parameter(6, N) # Actuation force
    b = opti.parameter(6, 1) # External force
    C = opti.parameter(const.N_FEET*3) # Contact flag
    S = opti.parameter(6, 6) # Weight matrix
    F_last = opti.parameter(const.N_FEET*3) # Previous GRFs
    F = opti.variable(const.N_FEET*3) # Decision variable - GRFs

    opti.minimize((A@F - b).T @ S @ (A@F - b) + PENALTY_L2_GRF * ca.sumsqr(F) + PENALTY_L2_GRF_SLEW * ca.sumsqr(F - F_last))
    opti.subject_to(C * F >= F_MIN)
    opti.subject_to(C * F <= F_MAX)

    m = model.body("trunk").subtreemass*0.9

    G = dynamics.get_G()

    K_P = np.diag([
        100, 100, 100,
        1000, 1000, 1000
    ])*-1
    K_D = np.diag([
        # 1, 1, 1,
        # 1, 1, 1,
        0, 0, 0,
        0, 0, 0
    ])*-1
    K = np.block([-K_P, -K_D])

    S_ = np.diag([
        1, 1, 1,
        1, 1, 1
    ])
    opti.set_value(S, S_)

    def update_opt(
        p_c, theta, p_c_dot, w_b, 
        F_last_: np.ndarray, 
        M_: np.ndarray,
        A_: np.ndarray
    ):
        # Rotation matrix from world frame to body frame from state euler angles
        b_R_w = math_utils.rpy_to_rot_mat(*theta)

        p_c_des = np.array([0.0223, 0.002, 0.2895])

        des_pitch = 0
        # des_pitch = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])

        des_roll = 0
        des_roll = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])

        des_yaw = 0
        # des_yaw = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])

        theta_des = np.array([des_roll, des_pitch, des_yaw])
        R_des = np.array(math_utils.rpy_to_rot_mat(*theta_des))
        p_c_dot_des = np.array([0, 0, 0])
        w_b_des = np.array([0, 0, 0])
        e = np.hstack([
            p_c - p_c_des,
            math_utils.log_so3(R_des @ b_R_w).flatten(),
            p_c_dot - p_c_dot_des,
            w_b - w_b_des,
        ]).reshape(12, 1)
        u = K @ e
        b_ = M_ @ (u + G)

        for i, name in enumerate(["u/lin/x", "u/lin/y", "u/lin/z", "u/ang/x", "u/ang/y", "u/ang/z"]):
            rr.log(name, rr.Scalar(b_[i]))
        for i, name in enumerate([
            "error/pos/lin/x", "error/pos/lin/y", "error/pos/lin/z",
            "error/pos/ang/x", "error/pos/ang/y", "error/pos/ang/z",
            "error/vel/lin/x", "error/vel/lin/y", "error/vel/lin/z",
            "error/vel/ang/x", "error/vel/ang/y", "error/vel/ang/z",
        ]):
            rr.log(name, rr.Scalar(-e[i]))

        contact_flags = [data.sensor(const.FEET_CONTACT_SENSOR_NAMES[n]).data > 0.0 for n in const.LEG_NAMES]
        C_ = np.concatenate([np.ones(3)*f for f in contact_flags])

        # opti.set_value(A, ca.DM(A_.tolist()))
        # opti.set_value(b, ca.DM(b_.tolist()))
        # opti.set_value(C, ca.DM(C_.tolist()))
        # opti.set_value(F_last, ca.DM(F_last_.tolist()))
        opti.set_value(A, A_)
        opti.set_value(b, b_)
        opti.set_value(C, C_)
        opti.set_value(F_last, F_last_)

    def ctrl_loop(model: MjModel, data: MjData, F_des: np.ndarray):
        J = []
        for leg_name, site_name in const.FEET_SITE_NAMES.items():
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, model.site(site_name).id)
            J.append(jacp)
        J = np.vstack(J)

        data.ctrl[:] = (J.T @ F_des)[6:]

    rr.init("adaptive_mpc", spawn=True, default_enabled=True)
    rr.set_time_seconds("mj_time", 0)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"{model.opt.timestep=}")

        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)

        sol = None
        solve_rate = 500
        solve_period = 1.0/solve_rate
        if solve_period < model.opt.timestep:
            raise ValueError(f"Solve rate is too high! {solve_period} < {model.opt.timestep}")
        solve_decimation = int(solve_period / model.opt.timestep)
        print(f"{solve_decimation=}")

        F_des = np.zeros((const.N_FEET*3))
        # opti.solver("ipopt", {
        #     'ipopt.print_level': 0, 
        #     'print_time': 0, 
        #     'ipopt.sb': 'yes'
        # })
        # opti.solver("qpoases", dict(printLevel="none"))
        opti.solver("osqp", dict(), dict(verbose=False))

        solve_timer = Timer(
            PeriodicCallback(
                # lambda duration: print(f"Solve time: {duration}"),
                n=100
            ),
        )

        ctr = 0
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            rr.set_time_seconds("mj_time", data.time)

            p_c = data.sensor("trunk_com_pos").data
            theta = math_utils.quat_to_rpy(data.sensor("trunk_body_quat").data)
            p_c_dot = data.sensor("trunk_com_linvel").data
            w_b = data.sensor("trunk_com_angvel").data
            p_i = [data.site(const.FEET_SITE_NAMES[n]).xpos for n in const.LEG_NAMES]

            # Rotation matrix from world frame to body frame from state euler angles
            b_R_w = math_utils.rpy_to_rot_mat(*theta)

            # Grab diagonal of inertia matrix and reshape to 3x3 matrix
            I_i = np.diag(model.body("trunk").inertia)

            # Rotate inertia matrix to body frame
            i_R_b = math_utils.quat_to_rot_mat(model.body("trunk").iquat)
            I_b = i_R_b.T @ I_i @ i_R_b
            I_G = b_R_w.T @ I_b @ b_R_w

            M_ = dynamics.get_M(m, I_G)
            A_ = dynamics.get_A(p_c, p_i)

            if ctr % solve_decimation == 0:
                with solve_timer:
                    update_opt(p_c, theta, p_c_dot, w_b, F_des, M_, A_)
                    opti.set_initial(F, F_des)
                    sol = opti.solve()
                    F_des = sol.value(F)

            
            X_hat = np.hstack([p_c, theta, p_c_dot, w_b])
            F_hat = F_des # TODO
            D = dynamics.get_D(*theta)
            H_bar = dynamics.get_H(M_, A_)
            B = dynamics.get_B()
            G = dynamics.get_G()
            
            for leg_idx, leg_name in enumerate(const.LEG_NAMES):
                for ax_idx, ax_name in enumerate(["x", "y", "z"]):
                    rr.log(f"des_grf/{leg_name}/{ax_name}", rr.Scalar(F_des[leg_idx*3+ax_idx]))

            ctr += 1

            ctrl_loop(model, data, F_des)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            viewer.sync()

            time_elapsed = time.time() - step_start
            rtf = time_elapsed/model.opt.timestep
            if ctr % 100 == 0:
                print(f"{rtf=}")
            rr.log("rtf", rr.Scalar(rtf))
            # if time_elapsed > model.opt.timestep:
            #     print(f"Overrun! RTF: {time_elapsed/model.opt.timestep}") 
            # time_until_next_step = model.opt.timestep - time_elapsed
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()