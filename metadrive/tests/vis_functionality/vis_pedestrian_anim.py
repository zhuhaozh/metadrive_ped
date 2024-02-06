from metadrive.envs.metadrive_env import MetaDriveEnv
# from metadrive.policy.idm_policy import IDMPolicy
# from metadrive.utils import setup_logger
# from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.traffic_participants.pedestrian_navi import PedestrianNavigation
from metadrive.component.traffic_participants.cyclist import Cyclist


if __name__ == "__main__":
    # setup_logger(True)
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0,
            "traffic_mode": "hybrid",
            "start_seed": 22,
            "accident_prob": 1.0,
            # "_disable_detector_mask":True,
            "debug_physics_world": False,
            "debug_static_world": False,
            "debug": False,
            # "global_light": True,
            # "cull_scene": False,
            # "image_observation": True,
            # "controller": "joystick",
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "need_inverse_traffic": False,
            # "rgb_clip": True,
            "map": "XSS",
            # "agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": True,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "show_interface": True,
            "force_destroy": False,
            # "camera_dist": -1,
            # "camera_pitch": 30,
            # "camera_smooth": False,
            "camera_height": 2,
            # "window_size": (2400, 1600),
            "show_coordinates": True,

            "traffic_vehicle_config":dict(
                show_navi_mark=False,
                show_dest_mark=False,
                enable_reverse=False,
                show_lidar=True,
                show_lane_line_detector=True,
                show_side_detector=True,
            ),

            "vehicle_config": {
                "enable_reverse": False,
            },
        }
    )
    
    import time

    start = time.time()
    o, _ = env.reset()

    env.switch_to_third_person_view()
    
    obj_1 = env.engine.spawn_object(PedestrianNavigation, position=[15, 5], heading_theta=0, random_seed=1, set_friction=True) # control by 
    obj_1.set_heading_theta(90, in_rad=False)
    obj_1.set_heading_theta(180, in_rad=False)
    obj_1.set_velocity(value=2, in_local_frame=True)
    

    env.vehicle.set_velocity([1, 0], in_local_frame=False)
    
    def clamp(i, mn=-1, mx=1):
        return min(max(i, mn), mx)

    for s in range(1, 10000):
        smp = env.action_space.sample()
        o, r, tm, tc, info = env.step(smp)
        # print(obj_1.actor.getJoints())
        # [root, pelvis, left_hip, left_knee, left_ankle, left_foot, left_foot_end, right_hip, right_knee, right_ankle, right_foot, 
        #  right_foot_end, spine1, spine2, spine3, neck, head, jaw, jaw_end, left_eye_smplhf, left_eye_smplhf_end, right_eye_smplhf, 
        #  right_eye_smplhf_end, left_collar, left_shoulder, left_elbow, left_wrist, left_index1, left_index2, left_index3, left_index3_end, 
        #  left_middle1, left_middle2, left_middle3, left_middle3_end, left_pinky1, left_pinky2, left_pinky3, left_pinky3_end, left_ring1, left_ring2, left_ring3, left_ring3_end, left_thumb1, left_thumb2, left_thumb3, left_thumb3_end, right_collar, right_shoulder, right_elbow, right_wrist, right_index1, right_index2, right_index3, right_index3_end, right_middle1, right_middle2, right_middle3, right_middle3_end, right_pinky1, right_pinky2, right_pinky3, right_pinky3_end, right_ring1, right_ring2, right_ring3, right_ring3_end, right_thumb1, right_thumb2, right_thumb3, right_thumb3_end]
        # obj_1.move([0, 1], 0.1)
        # if s == 10:
        #     obj_1.set_heading_theta(90, in_rad=False)


        obj_1.joints['left_shoulder'].setR(clamp((s - 100) / 100) * 120) # in degree
        obj_1.set_on_ground()
        env.render(
            text={
                "heading_diff": env.vehicle.heading_diff(env.vehicle.lane),
                "lane_width": env.vehicle.lane.width,
                "lateral": env.vehicle.lane.local_coordinates(env.vehicle.position),
                "current_seed": env.current_seed,
                "step": s,
            }
        )