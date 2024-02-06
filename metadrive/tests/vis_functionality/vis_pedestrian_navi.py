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
                # "show_lidar": True
                # "image_source": "depth_camera",
                # "random_color": True
                # "show_lidar": True,
                # "spawn_lane_index":("1r1_0_", "1r1_1_", 0),
                # "destination":"2R1_3_",
                # "show_side_detector": True,
                # "show_lane_line_detector": True,
                # "side_detector": dict(num_lasers=2, distance=50),
                # "lane_line_detector": dict(num_lasers=2, distance=50),
                # # "show_line_to_dest": True,
                # "show_dest_mark": True
            },
        }
    )
    
    import time

    start = time.time()
    o, _ = env.reset()

    env.switch_to_third_person_view()
    
    # 初始化
    obj_1 = env.engine.spawn_object(PedestrianNavigation, position=[10, 5], heading_theta=0, random_seed=1)
    obj_2 = env.engine.spawn_object(PedestrianNavigation, position=[25, 5], heading_theta=0, random_seed=1)
    # obj_3 = env.engine.spawn_object(PedestrianNavigation, position=[20, 6], heading_theta=0, random_seed=1)
    obj_4 = env.engine.spawn_object(PedestrianNavigation, position=[15, -3], heading_theta=0, random_seed=1, 
                                    set_friction=True) # control by 
    obj_4.set_heading_theta(90, in_rad=False)
    
    # cyc_1 = env.engine.spawn_object(Cyclist, position=[30, 6], heading_theta=0, random_seed=1)
    # cyc_1.set_velocity(direction=[1,0])

    env.vehicle.set_velocity([1, 0], in_local_frame=False)


    for s in range(1, 10000):
        smp = env.action_space.sample()
        o, r, tm, tc, info = env.step(smp)
        
        obj_4.move([0, 1], 0.1)
        if s == 10:
            # obj_1.set_angular_velocity(90, in_rad=False)
            obj_1.set_heading_theta(90, in_rad=False)
            obj_1.set_velocity(value=2, in_local_frame=True)

            # obj_4.set_heading_theta(90, in_rad=False)
            # obj_4.set_velocity(value=4, in_local_frame=True)
    
        elif s == 50:
            obj_1.set_heading_theta(0, in_rad=False)
            obj_1.set_velocity(value=1, in_local_frame=True)
        elif s == 75:
            obj_1.set_velocity(value=0, in_local_frame=True)

        elif s == 100:
            obj_1.set_velocity(value=2, in_local_frame=True)
        
        elif s == 100:
            obj_1.set_velocity(value=4, in_local_frame=True)

        obj_2.follow(obj_1)

        obj_2.get_posible_hit_obstacles()


        # print("------------------getZ-------------------")
        # print("obj_1: {:.4f}".format(obj_1.origin.getZ()))
        # print("obj_4: {:.4f}".format(obj_4.origin.getZ()))
        # check_restults = obj_4._state_check(debug=True)
        # for c in check_restults:
        #     if c[0] == "ROAD_EDGE_SIDEWALK":
        #         # print("obj_4: ", obj_4._state_check(debug=True))
        #         print("obj_4: ", check_restults)
        #         break
        # print("-----------------------------------------")

        obj_1.do_obstacle_aviodance()
        
        obj_1.set_on_ground()
        obj_2.set_on_ground()
        obj_4.set_on_ground()
        # obj_2.set_on_ground()
        # obj_1.get_nearest_obstacle()

        env.render(
            text={
                "heading_diff": env.vehicle.heading_diff(env.vehicle.lane),
                "lane_width": env.vehicle.lane.width,
                "lateral": env.vehicle.lane.local_coordinates(env.vehicle.position),
                "current_seed": env.current_seed,
                "step": s,
            }
        )