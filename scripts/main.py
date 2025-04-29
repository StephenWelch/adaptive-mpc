import mujoco
import mujoco.viewer
import casadi as ca
import time
import scipy
import rerun as rr
import numpy as np
from math_utils import skew, quat_to_rpy, euler_to_rotation_matrix, rotation_matrix_to_euler, log_map, get_R
from sim_utils import compute_site_jacobian
        
from casadi import vec

np.set_printoptions(precision=3, linewidth=1000, floatmode="fixed")

def main():
    
    model = mujoco.MjModel.from_xml_path("assets/unitree_go1/scene.xml")
    data = mujoco.MjData(model)

    N_FEET = 4
    N = 12
    M = N_FEET * 3
    LEG_NAMES = ["FR", "FL", "RR", "RL"]
    FEET_SITE_NAMES = {
        name: name 
        for name in LEG_NAMES
    }
    FEET_CONTACT_SENSOR_NAMES = {
        name: f"{name}_contact"
        for name in LEG_NAMES
    }
    LEG_JNT_IDXS = {
        name: [
            # Offset by +6 to account for FB DoFs, -1 to account for root joint ID
            5 + model.joint(f"{name}_hip_joint").id,
            5 + model.joint(f"{name}_thigh_joint").id,
            5 + model.joint(f"{name}_calf_joint").id,
        ]
        for name in LEG_NAMES
    }
    LEG_ACT_IDXS = {
        name: [
            model.actuator(f"{name}_hip").id,
            model.actuator(f"{name}_thigh").id,
            model.actuator(f"{name}_calf").id,
        ]
        for name in LEG_NAMES
    }

    opti = ca.Opti()

    PENALTY_L2_GRF = 0.0
    PENALTY_L2_GRF_SLEW = 0.0
    F_MIN = -2000.0
    F_MAX = 2000.0
    A = opti.parameter(6, N) # Actuation force
    b = opti.parameter(6, 1) # External force
    C = opti.parameter(N_FEET, N_FEET) # Contact flag
    S = opti.parameter(6, 6) # Weight matrix
    F_last = opti.parameter(N_FEET*3) # Previous GRFs
    F = opti.variable(N_FEET*3) # Decision variable - GRFs

    opti.minimize((A@F - b).T @ S @ (A@F - b) + PENALTY_L2_GRF * ca.sumsqr(F) + PENALTY_L2_GRF_SLEW * ca.sumsqr(F - F_last))
    opti.subject_to(vec(F) >= F_MIN)
    opti.subject_to(vec(F) <= F_MAX)

    def update_opt(model: mujoco.MjModel, data: mujoco.MjData, sol_last: ca.OptiSol | None = None):
        m = model.body("trunk").mass
        g = np.array([0, 0, -9.81]).reshape(3, 1)
        I_G = data.body("trunk").ximat.reshape(3, 3)

        R = data.body("trunk").xmat.reshape(3, 3)
        theta = quat_to_rpy(data.qpos[3:7])
        p_c = data.body("trunk").xipos
        p_i = np.array([data.site(FEET_SITE_NAMES[n]).xpos for n in LEG_NAMES])
        # p_c_dot = R.T @ data.body("trunk").cvel[:3]
        # w_b = R.T @ data.body("trunk").cvel[3:]
        p_c_dot = data.qvel[7:10]
        w_b = R.T @ data.qvel[10:13]
        # p_c_ddot = data.body("trunk").cacc[:3, None]
        # w_b_ddot = R.T @ data.body("trunk").cacc[3:, None]

        M = np.block([
            [m*np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), I_G],
        ])
        G = np.block([
            [g],
            [np.zeros((3, 1))],
        ])
        A_ = np.block([
            [np.eye(3) for _ in range(N_FEET)],
            [skew(p_i[i] - p_c) for i in range(N_FEET)] # TODO vectorize
        ])

        # b_ = M @ np.block([
        #     [p_c_ddot],
        #     [w_b_ddot],
        # ]) + G
        K_P = np.diag([
            500, 500, 500,
            # 50, 50, 50
            -150, 150, 150
        ])*-1
        K_D = np.diag([
            # 50, 50, 50,
            0, 0, 0,
            # 0, 1000, 1000,
            # -200, -200, -200,
            0, 0, 0
        ])*-1
        p_c_des = np.array([0.022, 0, 0.278])
        theta_des = np.array([0, np.deg2rad(10)*np.sin(2*np.pi*data.time), 0])
        theta_des = np.zeros(3)
        # R_des = get_R(*theta_des)
        R_des = np.array(euler_to_rotation_matrix(*theta_des))
        p_c_dot_des = np.array([0, 0, 0])
        w_b_des = np.array([0, 0, 0])
        K = np.block([-K_P, -K_D])
        e = np.hstack([
            p_c - p_c_des,
            # log_map(R_des @ get_R(*quat_to_rpy(data.qpos[3:7])).T).flatten(),
            # scipy.linalg.logm(R_des @ get_R(*quat_to_rpy(data.qpos[3:7])).T).flatten(),
            log_map(R_des @ R.T).flatten(),
            # rotation_log(R_des @ R.T).flatten(),
            p_c_dot - p_c_dot_des,
            w_b - w_b_des,
        ]).reshape(12, 1)
        u = K @ e
        b_ = M @ (u + G)
        # b_ = M @ G

        # print(f"{np.rad2deg(theta)=}")
        # print(f"{R_des=}")
        # print(f"{R=}")
        # print(f"theta: {np.rad2deg(theta)}")
        # print(f"theta_des: {np.rad2deg(theta_des)}")
        # print(f"rot_err: {log_map(R_des @ R.T)}")
        # print(f"{p_c}")
        # print(f"Error: {e}")
        # print(f"{u[3:]=}")
        for i, name in enumerate(["u/lin/x", "u/lin/y", "u/lin/z", "u/ang/x", "u/ang/y", "u/ang/z"]):
            rr.log(name, rr.Scalar(u[i]))
        for i, name in enumerate([
            "error/pos/lin/x", "error/pos/lin/y", "error/pos/lin/z",
            "error/pos/ang/x", "error/pos/ang/y", "error/pos/ang/z",
            "error/vel/lin/x", "error/vel/lin/y", "error/vel/lin/z",
            "error/vel/ang/x", "error/vel/ang/y", "error/vel/ang/z",
        ]):
            rr.log(name, rr.Scalar(-e[i]))

        C_ = np.diag([data.sensor(FEET_CONTACT_SENSOR_NAMES[n]).data > 0.0 for n in LEG_NAMES])
        S_ = np.diag([
            1, 1, 1,
            10, 10, 10
        ])
        if sol_last is not None:
            F_last_ = sol_last.value(F_last)
        else:
            F_last_ = np.zeros((N_FEET*3))

        opti.set_value(A, ca.DM(A_.tolist()))
        opti.set_value(b, ca.DM(b_.tolist()))
        opti.set_value(C, ca.DM(C_.tolist()))
        opti.set_value(S, ca.DM(S_.tolist()))
        opti.set_value(F_last, ca.DM(F_last_.tolist()))

    def ctrl_loop(model: mujoco.MjModel, data: mujoco.MjData, F_des: np.ndarray):
        for foot_idx, leg_name in enumerate(LEG_NAMES):
            site_name = FEET_SITE_NAMES[leg_name]
            act_idxs = LEG_ACT_IDXS[leg_name]
            jnt_idxs = LEG_JNT_IDXS[leg_name]

            J = compute_site_jacobian(model, data, model.site(site_name).id)[:3, jnt_idxs]
            F = F_des[foot_idx*3:foot_idx*3+3]

            data.ctrl[act_idxs] = J.T @ F
            
        # print(data.ctrl)

    rr.init("adaptive_mpc", spawn=True)
    rr.set_time_seconds("mj_time", 0)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        print(f"{model.opt.timestep=}")

        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
        # init_pos = np.array([model.site(FEET_SITE_NAMES[n]).pos for n in LEG_NAMES])

        sol = None
        opti.solver("ipopt", {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            rr.set_time_seconds("mj_time", data.time)
            update_opt(model, data, sol)
            if sol is not None:
                opti.set_initial(F, F_des)
            sol = opti.solve()
            F_des = sol.value(F)

            # feet_err = init_pos - np.array([model.site(FEET_SITE_NAMES[n]).pos for n in LEG_NAMES])
            # F_des = np.array([
            #     [0, 0, 0], # FR
            #     [0, 0, 0], # FL
            #     [0, 0, 0], # RR
            #     [0, 0, 1], # RL
            # ]).flatten() * -100
            ctrl_loop(model, data, F_des)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # input()

            # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

    ...

if __name__ == "__main__":
    main()