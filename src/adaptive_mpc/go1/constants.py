"""Constants used throughout the adaptive MPC system."""

from mujoco._structs import MjModel, MjData

class Go1Constants:
    def __init__(self, model: MjModel):
        # Number of feet and related dimensions
        self.N_FEET = 4

        # Leg names and related mappings
        self.LEG_NAMES = ["FR", "FL", "RR", "RL"]
        self.FEET_SITE_NAMES = {
            name: name 
            for name in self.LEG_NAMES
        }
        self.FEET_CONTACT_SENSOR_NAMES = {
            name: f"{name}_contact"
            for name in self.LEG_NAMES
        }

        # Useful indices
        self.LEG_JNT_IDXS = {
            name: [
                # Offset by +6 to account for FB DoFs, -1 to account for root joint ID
                5 + model.joint(f"{name}_hip_joint").id,
                5 + model.joint(f"{name}_thigh_joint").id,
                5 + model.joint(f"{name}_calf_joint").id,
            ]
            for name in self.LEG_NAMES
        }
        self.LEG_ACT_IDXS = {
            name: [
                model.actuator(f"{name}_hip").id,
                model.actuator(f"{name}_thigh").id,
                model.actuator(f"{name}_calf").id,
            ]
            for name in self.LEG_NAMES
        }