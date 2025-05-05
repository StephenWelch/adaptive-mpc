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
from adaptive_mpc import srb_control

np.set_printoptions(precision=3, linewidth=1000, floatmode="fixed")

def main():
    model = mujoco.MjModel.from_xml_path("assets/unitree_go1/scene.xml")
    data = mujoco.MjData(model)

    const = Go1Constants(model)

    # Compute trunk inertia in body frame
    I_i = np.diag(model.body("trunk").inertia)
    i_R_b = math_utils.quat_to_rot_mat(model.body("trunk").iquat)
    I_b = i_R_b.T @ I_i @ i_R_b

    # Set SRB mass to 90% of *total* robot mass, since legs are heavy in reality
    m = model.body("trunk").subtreemass*0.9
    G = dynamics.get_G()

    # PD gains for baseline controller
    K_P = -np.diag([
        100, 100, 100,
        1000, 1000, 1000
    ])
    K_D = -np.diag([
        # 1, 1, 1,
        # 1, 1, 1,
        0, 0, 0,
        0, 0, 0
    ])
    K = np.block([-K_P, -K_D])

    def ctrl_loop(model: MjModel, data: MjData, F_des: np.ndarray):
        """Computes joint torques from desired GRFs using end-effector Jacobians

        Args:
            model (MjModel): MuJoCo model
            data (MjData): MuJoCo data
            F_des (np.ndarray): Desired GRFs
        """
        J = []
        for leg_name, site_name in const.FEET_SITE_NAMES.items():
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, model.site(site_name).id)
            J.append(jacp)
        J = np.vstack(J)

        data.ctrl[:] = (J.T @ F_des)[6:]

    rr.init("adaptive_mpc", spawn=True, default_enabled=True)
    rr.set_time_seconds("mj_time", 0)

    solve_rate = 500
    solve_period = 1.0/solve_rate
    if solve_period < model.opt.timestep:
        raise ValueError(f"Solve rate is too high! {solve_period} < {model.opt.timestep}")
    solve_decimation = int(solve_period / model.opt.timestep)
    print(f"{solve_decimation=}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"{model.opt.timestep=}")

        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)

        srb_qp = srb_control.SingleRigidBodyQP(const.N_FEET)
        ref_srb_qp = srb_control.SingleRigidBodyQP(const.N_FEET)

        solve_timer = Timer(
            PeriodicCallback(
                lambda duration: print(f"Solve time: {duration}"),
                n=100
            ),
        )

        ctr = 0
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            rr.set_time_seconds("mj_time", data.time)

            # Get SRB state
            p_c = data.sensor("trunk_com_pos").data
            theta = math_utils.quat_to_rpy(data.sensor("trunk_body_quat").data)
            p_c_dot = data.sensor("trunk_com_linvel").data
            w_b = data.sensor("trunk_com_angvel").data
            p_i = [data.site(const.FEET_SITE_NAMES[n]).xpos for n in const.LEG_NAMES]
            contact_flags = [data.sensor(const.FEET_CONTACT_SENSOR_NAMES[n]).data > 0.0 for n in const.LEG_NAMES]

            # Get desired SRB state
            p_c_des = np.array([0.0223, 0.002, 0.2895])
            des_pitch = 0
            # des_pitch = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])
            des_roll = 0
            # des_roll = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])
            des_yaw = 0
            # des_yaw = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])
            theta_des = np.array([des_roll, des_pitch, des_yaw])
            R_des = np.array(math_utils.rpy_to_rot_mat(*theta_des))
            p_c_dot_des = np.array([0, 0, 0])
            w_b_des = np.array([0, 0, 0])

            # Compute SRB dynamics info
            b_R_w = math_utils.rpy_to_rot_mat(*theta)
            I_G = b_R_w.T @ I_b @ b_R_w
            M = dynamics.get_M(m, I_G)
            A = dynamics.get_A(p_c, p_i)

            # Get baseline control
            e = srb_control.compute_error(p_c, theta, p_c_dot, w_b, p_c_des, theta_des, p_c_dot_des, w_b_des)
            u = K @ e
            b_d = M @ (u + G)
            
            if ctr % solve_decimation == 0:
                with solve_timer:
                    F_des = srb_qp(p_c, theta, p_c_dot, w_b, M, A, b_d, contact_flags)
            
            # X_hat = np.hstack([p_c, theta, p_c_dot, w_b])
            # F_hat = F_des # TODO
            # D = dynamics.get_D(*theta)
            # H_bar = dynamics.get_H(M_, A_)
            # B = dynamics.get_B()
            # G = dynamics.get_G()
            
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