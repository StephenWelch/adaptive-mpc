import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/unitree_go1/scene.xml")


data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(data.body("trunk"))
    print(data.site("FR"))
    print(data.qpos[:7])
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()