from metadrive.envs import MetaDriveEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils import generate_gif
# from IPython.display import Image


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from functools import partial
# from IPython.display import clear_output
import os


def create_env(need_monitor=False):
    env = MetaDriveEnv(dict(map="C",
                      # This policy setting simplifies the task                      
                      discrete_action=False,
                      discrete_throttle_dim=3,
                      discrete_steering_dim=3,
                      horizon=500,
                      # scenario setting
                      random_spawn_lane_index=False,
                      num_scenarios=1,
                      start_seed=5,
                      traffic_density=0,
                      accident_prob=0,
                      log_level=50))
    if need_monitor:
        env = Monitor(env)
    return env


if __name__ == '__main__':
    # env=create_env()
    # env.reset()

    # ret = env.render(mode="topdown", 
    #                 window=False,
    #                 screen_size=(600, 600), 
    #                 camera_position=(50, 50))
    # env.close()

    # plt.axis("off")
    # plt.imshow(ret)
    # plt.savefig("a.jpg")


    # set_random_seed(0)
    # # 4 subprocess to rollout
    train_env=SubprocVecEnv([partial(create_env, True) for _ in range(4)]) 
    model = PPO("MlpPolicy", 
                train_env,
                n_steps=4096,
                verbose=1)
    
    # # train = False
    model.learn(total_timesteps=300_000,
                log_interval=1)

    # # else:

    # evaluation
    total_reward = 0
    env=create_env()
    obs, _ = env.reset()
    try:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            ret = env.render(mode="topdown", 
                            screen_record=True,
                            window=False,
                            screen_size=(600, 600), 
                            camera_position=(50, 50))
            if done:
                print("episode_reward", total_reward)
                break
                
        env.top_down_renderer.generate_gif()
    finally:
        env.close()

    print("gif generation is finished ...")