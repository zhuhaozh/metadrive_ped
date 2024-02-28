# from metadrive.envs import MetaDriveEnv

import torch
import numpy as np
import os.path as osp
from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.engine.logger import get_logger
from stable_baselines3 import PPO

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

