# from metadrive.envs import MetaDriveEnv

import torch
import numpy as np
import os.path as osp
from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.engine.logger import get_logger
from stable_baselines3 import PPO

from metadrive.utils.math import panda_vector

ckpt_path = osp.join(osp.dirname(__file__), "expert_weights.npz")
_expert_weights = None
_expert_observation = None

logger = get_logger()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model():
    # model = PPO.load("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/ppo_ped_c.zip", print_system_info=False)
    model = PPO.load("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/ppo_ped.zip", print_system_info=False)
    return model


def torch_expert(vehicle, deterministic=False, need_obs=False):
    global _expert_weights
    global _expert_observation
    expert_obs_cfg = dict(
        lidar=dict(num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0),
        random_agent_model=False
    )
    origin_obs_cfg = dict(
        lidar=dict(num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0),
        random_agent_model=False
    )

    with torch.no_grad():  # Disable gradient computation
        model = load_model()
        
        config = get_global_config().copy()
        config["vehicle_config"].update(expert_obs_cfg)
        _expert_observation = LidarStateObservation(config)

        obs = _expert_observation.observe(vehicle)

        obs = obs.reshape(1, -1)

        action, _states = model.predict(obs, deterministic=True)
        action = action.squeeze(0)  # Move back to CPU and remove batch dimension

    return (action, obs.cpu().numpy()) if need_obs else action

from panda3d.core import Point3, Vec2, LPoint3f
dests = [(10.0, 14.0), (63.5, -45.0), (8.0, 14.0), (10.0, 18.0), (18.0, 18.0)]
# dests = [(63.0 45.0), (8.0, 14.0), (10.0, 18.0), (18.0, 18.0)]
def rule_expert(vehicle, deterministic=False, need_obs=False):
    dest_pos = vehicle.navigation.get_checkpoints()[0]
    position = vehicle.position

    dest = panda_vector(dest_pos[0], dest_pos[1])
    vec_to_2d = dest - position 
    dist_to = vec_to_2d.length()

    heading = Vec2(*vehicle.heading).signedAngleDeg(vec_to_2d) * 3

    if dist_to > 2:
        vehicle._body.setAngularMovement(heading)
        vehicle._body.setLinearMovement(LPoint3f(0 , 1, 0) * 6, True)
    else:
        vehicle._body.setLinearMovement(LPoint3f(0 , 1, 0) * 1, True)
    return None

def get_dest_heading(obj, dest_pos):
    position = obj.position

    dest = panda_vector(dest_pos[0], dest_pos[1])
    vec_to_2d = dest - position 
    dist_to = vec_to_2d.length()

    heading = Vec2(*obj.heading).signedAngleDeg(vec_to_2d)
    return heading