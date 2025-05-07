from tracemalloc import start
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

np.set_printoptions(precision=3, linewidth=1000, floatmode="fixed", suppress=True)


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
    B = dynamics.get_B()
    G = dynamics.get_G()

    # PD gains for baseline controller
    # K_P = 0.1*np.diag([
    #     1000, 1000, 1000,
    #     1000, 1000, 1000
    # ])
    K_P = np.diag([
        1000, 1000, 1000,
        1000, 1000, 1000
    ])
    K_D = np.diag([
        100, 100, 100,
        100, 100, 100,
        # 0, 0, 0,
        # 0, 0, 0
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
        data.ctrl[:] = (J.T @ -F_des)[6:]

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
        mujoco.mj_step(model, data)

        srb_qp = srb_control.SingleRigidBodyQP(const.N_FEET)
        ref_srb_qp = srb_control.SingleRigidBodyQP(const.N_FEET)

        ref_model = dynamics.Go1ReferenceModel(model.opt.timestep)
        srb_l1 = srb_control.SingleRigidBodyL1Adaptation(model.opt.timestep, B, K)
        theta_hat = np.zeros((6,1))
        theta_hat_lpf = np.zeros((6,1))

        F_des = np.zeros((const.N_FEET*3,))
        F_des_ref = np.zeros((const.N_FEET*3,))

        u_pd = np.zeros((6,1))
        u_pd_ref = np.zeros((6,1))

        p_i = [data.site(const.FEET_SITE_NAMES[n]).xpos for n in const.LEG_NAMES]

        p_c = data.sensor("trunk_com_pos").data
        theta = math_utils.quat_to_rpy(data.sensor("trunk_body_quat").data)
        p_c_dot = data.sensor("trunk_com_linvel").data
        w_b = data.sensor("trunk_com_angvel").data

        p_c_ref = np.array([0.0223, 0.002, 0.2895])
        theta_ref = np.zeros((3,))
        p_c_dot_ref = np.zeros((3,))
        w_b_ref = np.zeros((3,))

        # Get desired SRB state
        p_c_des = np.array([0.0223, 0.002, 0.2895])
        des_pitch = 0
        # des_pitch = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])
        des_roll = 0
        # des_roll = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])
        des_yaw = 0
        # des_yaw = np.interp(np.sin(2*np.pi*0.5*data.time), [-1, 1], [np.deg2rad(-10), np.deg2rad(10)])
        theta_des = np.array([des_roll, des_pitch, des_yaw])
        p_c_dot_des = np.array([0, 0, 0])
        w_b_des = np.array([0, 0, 0])

        b_R_w = math_utils.rpy_to_rot_mat(*theta)
        I_G = b_R_w.T @ I_b @ b_R_w
        # m_scale = 0.25
        m_scale = 1.0
        M = dynamics.get_M(m*m_scale, I_G)
        A = dynamics.get_A(p_c, p_i)
        D = dynamics.get_D(*theta)

        b_R_w_ref = math_utils.rpy_to_rot_mat(*theta_ref)
        I_G_bar = b_R_w_ref.T @ I_b @ b_R_w_ref
        M_bar = dynamics.get_M(m*m_scale, I_G)
        A_bar = dynamics.get_A(p_c_ref, p_i)
        H_bar = dynamics.get_H(M_bar, A)
        D_bar = dynamics.get_D(*theta_ref)

        solve_timer = Timer(
            PeriodicCallback(
                lambda duration: print(f"Solve time: {duration}"),
                n=100
            ),
        )

        # input()
        ctr = 0
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Step sim
            mujoco.mj_step(model, data)
            rr.set_time_seconds("mj_time", data.time)

            # Get sim state
            p_c = data.sensor("trunk_com_pos").data
            theta = math_utils.quat_to_rpy(data.sensor("trunk_body_quat").data)
            p_c_dot = data.sensor("trunk_com_linvel").data
            w_b = data.sensor("trunk_com_angvel").data
            p_i = [data.site(const.FEET_SITE_NAMES[n]).xpos for n in const.LEG_NAMES]
            contact_flags = [data.sensor(const.FEET_CONTACT_SENSOR_NAMES[n]).data > 0.0 for n in const.LEG_NAMES]

            # Update SRB dynamics from sim state
            b_R_w = math_utils.rpy_to_rot_mat(*theta)
            # b_R_w = math_utils.R_z(theta_ref[2]).T
            I_G = b_R_w.T @ I_b @ b_R_w
            M = dynamics.get_M(m*m_scale, I_G)
            A = dynamics.get_A(p_c, p_i)

            # Get ref state
            p_c_ref, theta_ref, p_c_dot_ref, w_b_ref = ref_model(p_c_ref, theta_ref, p_c_dot_ref, w_b_ref, F_des_ref, u_pd_ref, theta_hat_lpf, theta_hat, D_bar, H_bar, B, G)

            # Update SRB dynamics from ref state
            b_R_w_ref = math_utils.rpy_to_rot_mat(*theta_ref)
            # b_R_w_ref = math_utils.R_z(theta_ref[2]).T
            I_G_bar = b_R_w_ref.T @ I_b @ b_R_w_ref
            M_bar = dynamics.get_M(m*m_scale, I_G_bar)
            A_bar = dynamics.get_A(p_c_ref, p_i)
            H_bar = dynamics.get_H(M_bar, A_bar)
            # D_bar = dynamics.get_D(*theta_ref)
            D_bar = dynamics.get_D(*theta)

            # Compute error
            e = srb_control.compute_error(p_c, theta, p_c_dot, w_b, p_c_des, theta_des, p_c_dot_des, w_b_des)
            e_hat = srb_control.compute_error(p_c_ref, theta_ref, p_c_dot_ref, w_b_ref, p_c_des, theta_des, p_c_dot_des, w_b_des)

            # Step adaptation law
            theta_hat, theta_hat_lpf = srb_l1(e, e_hat)

            # Get baseline control - accels
            u_pd = K @ e
            u_pd_ref = K @ e_hat
            
            # Accels -> Torques + Dynamic inversion
            b_d = M @ (u_pd + theta_hat_lpf - G)
            b_d_ref = M_bar @ (u_pd_ref + theta_hat - theta_hat_lpf - G)
            # b_d_ref = M_bar @ (u_pd_ref - G)
   
            if ctr % solve_decimation == 0:
                with solve_timer:
                    F_des = srb_qp(A, b_d, contact_flags)
                    F_des_ref = ref_srb_qp(A_bar, b_d_ref, contact_flags)

            # p_c_ref = p_c
            # theta_ref = theta
            # p_c_dot_ref = p_c_dot
            # w_b_ref = w_b        

            print(f"Ref model:")
            print(f"{np.linalg.norm(e_hat)=}")
            print(f"{p_c_ref=}")
            print(f"{theta_ref=}")
            print(f"{p_c_dot_ref=}")
            print(f"{w_b_ref=}")
            print(f"{b_d_ref=}")

            print(f"Real:")
            print(f"{np.linalg.norm(e)=}")
            print(f"{p_c=}")
            print(f"{theta=}")
            print(f"{p_c_dot=}")
            print(f"{w_b=}")
            print(f"{b_d=}")

            # scene = viewer.user_scn
            # scene.ngeom += 1
            # mujoco.mjv_initGeom(
            #     scene.geoms[scene.ngeom-1],
            #     mujoco.mjtGeom.mjGEOM_BOX,
            #     np.array([1, 1, 1]), # Size
            #     p_c_ref, # Position
            #     math_utils.rpy_to_rot_mat(*theta_ref), # Rotation
            #     np.array((0, 0, 0)).astype(np.float32), # Color
            # )

            for i, n in enumerate(["pos/x", "pos/y", "pos/z", "rot/x", "rot/y", "rot/z"]):
                rr.log(f"theta_hat/{n}", rr.Scalar(theta_hat[i]))
                rr.log(f"theta_hat_lpf/{n}", rr.Scalar(theta_hat_lpf[i]))
                rr.log(f"des_wrench/{n}", rr.Scalar(b_d[i]))
            for leg_idx, leg_name in enumerate(const.LEG_NAMES):
                for ax_idx, ax_name in enumerate(["x", "y", "z"]):
                    rr.log(f"des_grf/{leg_name}/{ax_name}", rr.Scalar(F_des[leg_idx*3+ax_idx]))
            for i, n in enumerate(["x", "y", "z"]):
                rr.log(f"theta/{n}", rr.Scalar(theta[i]))
                rr.log(f"theta_ref/{n}", rr.Scalar(theta_ref[i])) 
                rr.log(f"theta_des/{n}", rr.Scalar(theta_des[i])) 
            for i, n in enumerate(["x", "y", "z"]):
                rr.log(f"pos/{n}", rr.Scalar(p_c[i]))
                rr.log(f"pos_ref/{n}", rr.Scalar(p_c_ref[i])) 
                rr.log(f"pos_des/{n}", rr.Scalar(p_c_des[i])) 
            rr.log("err_norm", rr.Scalar(np.linalg.norm(e)))
            rr.log("err_norm_ref", rr.Scalar(np.linalg.norm(e_hat)))

            ctr += 1

            ctrl_loop(model, data, F_des)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            viewer.sync()

            # input()

            time_elapsed = time.time() - step_start
            rtf = time_elapsed/model.opt.timestep
            rr.log("rtf", rr.Scalar(rtf))
            if ctr % 100 == 0:
                print(f"{rtf=}")
            if data.time > 8:
                exit()
            # time_until_next_step = model.opt.timestep - time_elapsed
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()