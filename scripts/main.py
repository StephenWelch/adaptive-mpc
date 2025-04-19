import mujoco
import mujoco.viewer
import casadi
import time
import numpy as np
        
from casadi import vec

np.set_printoptions(precision=3, linewidth=1000)

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

    opti = casadi.Opti()

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

    def update_opt(model: mujoco.MjModel, data: mujoco.MjData):
        ...

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

        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            feet_err = init_pos - np.array([model.site(FEET_SITE_NAMES[n]).pos for n in LEG_NAMES])
            

            F_des = np.array([
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ]) * -100
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