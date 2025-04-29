import mujoco
import mujoco.viewer
import casadi as ca
import time
import numpy as np
        
from casadi import vec

np.set_printoptions(precision=3, linewidth=1000)

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

def quat_to_euler(q):
    return np.array([
        np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)),
        np.arcsin(2*(q[0]*q[2] - q[3]*q[1])),
        np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)),
    ])

def get_R(model: mujoco.MjModel, data: mujoco.MjData):
    roll, pitch, yaw = quat_to_euler(data.qpos[3:7])
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

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

    print(model.body("trunk").subtreemass)

    opti = ca.Opti()

    PENALTY_L1_GRF = 0.0
    PENALTY_L2_GRF_SLEW = 0.0
    F_MIN = -1000.0
    F_MAX = 1000.0
    A = opti.parameter(6, N) # Actuation force
    b = opti.parameter(6, 1) # External force
    C = opti.parameter(N_FEET, N_FEET) # Contact flag
    S = opti.parameter(6, 6) # Weight matrix
    F_last = opti.parameter(N_FEET, 3) # Previous GRFs
    F = opti.variable(N_FEET, 3) # Decision variable - GRFs

    opti.minimize((A@vec(F) - b).T @ S @ (A@vec(F) - b))
    opti.subject_to(vec(F) >= F_MIN)
    opti.subject_to(vec(F) <= F_MAX)

    def update_opt(model: mujoco.MjModel, data: mujoco.MjData, sol_last: ca.OptiSol | None = None):
        m = model.body("trunk").mass
        g = np.array([0, 0, -9.81]).reshape(3, 1)
        I_G = data.body("trunk").ximat.reshape(3, 3)

        R = data.body("trunk").xmat.reshape(3, 3)
        theta = quat_to_euler(data.qpos[3:7])
        p_c = data.body("trunk").xipos
        p_i = np.array([data.site(FEET_SITE_NAMES[n]).xpos for n in FEET_SITE_NAMES])
        p_c_dot = data.body("trunk").cvel[:3]
        w_b = data.body("trunk").cvel[3:]
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
        K_P = np.eye(6)
        K_D = np.eye(6)
        x_des = np.array([
            0, 0, 0.445, # p_c
            0, 0, 0, # theta
            0, 0, 0, # p_c_dot
            0, 0, 0, # w_dot
        ])
        x = np.block([p_c, theta, p_c_dot, w_b])
        K = np.block([-K_P, -K_D])
        u = K @ (x_des - x)
        b_ = M @ (u.reshape(6, 1) + G)

        C_ = np.eye(N_FEET)
        S_ = np.eye(6)
        if sol_last is not None:
            F_last_ = sol_last.value(F_last)
        else:
            F_last_ = np.zeros((N_FEET, 3))

        print(A_.shape)
        opti.set_value(A, ca.DM(A_.tolist()))
        opti.set_value(b, ca.DM(b_.tolist()))
        opti.set_value(C, ca.DM(C_.tolist()))
        opti.set_value(S, ca.DM(S_.tolist()))
        opti.set_value(F_last, ca.DM(F_last_.tolist()))

    def ctrl_loop(model: mujoco.MjModel, data: mujoco.MjData, F: np.ndarray):

        for foot_idx, leg_name in enumerate(LEG_NAMES):
            site_name = FEET_SITE_NAMES[leg_name]
            act_idxs = LEG_ACT_IDXS[leg_name]
            jnt_idxs = LEG_JNT_IDXS[leg_name]

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))

            mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site_name).id)
            jacp = jacp[:, jnt_idxs]
            # F assumed to be in hip frame
            data.ctrl[act_idxs] = jacp.T @ F[foot_idx]
            
        print(data.ctrl)


    with mujoco.viewer.launch_passive(model, data) as viewer:

        init_pos = np.array([model.site(FEET_SITE_NAMES[n]).pos for n in LEG_NAMES])
        print(init_pos)

        sol = None
        opti.solver("ipopt")

        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # feet_err = init_pos - np.array([model.site(FEET_SITE_NAMES[n]).pos for n in LEG_NAMES])
            

            update_opt(model, data, sol)
            sol = opti.solve()
            F_des = sol.value(F)
            # F_des = np.array([
            #     [0, 0, 1],
            #     [0, 0, 1],
            #     [0, 0, 1],
            #     [0, 0, 1],
            # ]) * -100
            ctrl_loop(model, data, F_des)
            mujoco.mj_step(model, data)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    ...

if __name__ == "__main__":
    main()