from metadrive.component.agents.pedestrian.pedestrian_type import SimplePedestrian
from metadrive.envs.metadrive_env import MetaDriveEnv

from panda3d.core import LVecBase3f

from metadrive.tests.bvh import Bvh
from metadrive.tests.load_egg import load_egg



def set_motions(obj, motion):
    # print(len(motion))
    for joint_name in motion.keys():
        joint = obj.actor.controlJoint(None, "modelRoot", joint_name)

        rotation =  motion[joint_name]['rotation']
        position = motion[joint_name]['position']

        delta_xyz = LVecBase3f(*position)
        if rotation is None:
            joint.setPos(delta_xyz)
        else:
            r, p, h = rotation
            delta_hpr = LVecBase3f(h, p, r)
            joint.setPosHpr(delta_xyz, delta_hpr)


def set_motion(obj, joint_name, rotation, offset=None):
    joint = obj.actor.controlJoint(None, "modelRoot", joint_name)

    r, p, h = rotation
    delta_hpr = LVecBase3f(h, p, r)
    if offset is None:
        joint.setHpr(delta_hpr)
    else:
        offset = LVecBase3f(*offset)
        joint.setPosHpr(offset, delta_hpr)


if __name__ == "__main__":
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
            "show_coordinates": False,

            "traffic_vehicle_config":dict(
                show_navi_mark=False,
                show_dest_mark=False,
                enable_reverse=False,
                show_lidar=False,
                show_lane_line_detector=False,
                show_side_detector=False,
            ),

            "vehicle_config": {
                "enable_reverse": True,
            },
        }
    )

    o, _ = env.reset()

    env.switch_to_third_person_view()

    traffic_vehicle_config1=dict(spawn_position_heading=[(10, 7), 0])
    obj_1 = env.engine.spawn_object(SimplePedestrian, vehicle_config=traffic_vehicle_config1) # control by 
    # obj_1.actor.loop('test')


    traffic_vehicle_config2=dict(spawn_position_heading=[(10, 3), 0])
    obj_2 = env.engine.spawn_object(SimplePedestrian, vehicle_config=traffic_vehicle_config2) # control by 
    
    # load bvh file
    with open('./metadrive/tests/vis_functionality/motion_files/walk60.bvh') as f:
        mocap = Bvh(f.read())
    bvh_motions = mocap.get_all_motions(abs=False)
    joint_names = mocap.get_joints_names()

    # load egg file
    egg_motions = load_egg("./metadrive/tests/vis_functionality/motion_files/walk60.egg")

    degree_lst = list(range(0, 70, 4)) + list(range(70, -70, -4)) + list(range(-70, 0, 4))

    for s in range(0, 10000):
        smp = env.action_space.sample()
        o, r, tm, tc, info = env.step(smp)

        """
        for joint_name in joint_names:
            # joint = mocap.get_joint(joint_name)
            # offset = mocap.joint_offset(joint_name)
            channels = mocap.joint_channels(joint_name)
            Xposition, Yposition, Zposition, Xrotation, Yrotation, Zrotation \
                = mocap.frame_joint_channels(s % 50, joint_name, channels)

            offset = (Xposition, Yposition, Zposition) # if s % 50 == 0 else None
            rotation = Xrotation, Yrotation, Zrotation
            set_motion(obj_2, joint_name, rotation, offset)
        """

        motion1 = bvh_motions[s % 50] # bvh
        set_motions(obj_1, motion1)

        motion2 = egg_motions[s % 50] # egg
        set_motions(obj_2, motion2)

        env.render(
            text={
                "heading_diff": env.vehicle.heading_diff(env.vehicle.lane),
                "lane_width": env.vehicle.lane.width,
                "lateral": env.vehicle.lane.local_coordinates(env.vehicle.position),
                "current_seed": env.current_seed,
                "step": s,
            }
        )