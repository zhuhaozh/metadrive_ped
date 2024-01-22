from direct.actor.Actor import Actor
from panda3d.bullet import BulletCylinderShape
from panda3d.core import LVector3
from panda3d.core import CollisionTraverser, CollisionNode
# from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.core import Point3, Vec2, LPoint3f, Material

from panda3d.core import CollisionHandlerQueue, CollideMask, CollisionRay
from metadrive.component.road_network.node_road_network import NodeRoadNetwork

from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.component.navigation_module.node_network_navigation import PedestrianNodeNetworkNavigation, NodeNetworkNavigation

from metadrive.constants import CollisionGroup
from metadrive.constants import MetaDriveType, AssetPaths

from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math import norm
from metadrive.component.sensors.lidar import Lidar
# from metadrive.component.vehicle_module.lidar import Lidar
# from metadrive.component.sensors.distance_detector import DistanceDetector as GroundDetector

from metadrive.engine.physics_node import BaseRigidBodyNode, BaseCharacterControllerNode

from metadrive.utils.coordinates_shift import panda_vector, metadrive_vector, panda_heading
import numpy as np
from metadrive.engine.engine_utils import get_engine
from metadrive.utils import Config, safe_clip_for_small_array

import random


class PedestrianBase(BaseTrafficParticipant):
    MASS = 70  # kg
    TYPE_NAME = MetaDriveType.PEDESTRIAN
    COLLISION_MASK = CollisionGroup.TrafficParticipants

    RADIUS = 0.35
    # HEIGHT = 1.75

    STATES = ['walk', "run", "idle"]

    def __init__(self, position, heading_theta, random_seed=None, name=None, set_friction=None):
        rand_texture = AssetPaths.Pedestrian.get_random_texture()
        rand_texture = AssetPaths.Pedestrian.PEDESTRIAN_TEXTURE["0"]
        self.HEIGHT = rand_texture['height']
        # print("rand_texture", rand_texture)

        super(PedestrianBase, self).__init__(
            position, heading_theta, random_seed, name=name)
        n = BaseRigidBodyNode(self.name, self.TYPE_NAME)
        if set_friction:
            self.add_body(n, friction=100, anisotropic_friction=(5., 5., 5.))
        else:
            self.add_body(n)

        self._body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self._body.setDeactivationEnabled(False)

        """
        # shape = BulletCylinderShape(self.RADIUS, self.HEIGHT)
        # n = BaseCharacterControllerNode(self.name, self.TYPE_NAME, shape=shape)
        # self.add_body(n)
        # self._body.setDeactivationEnabled(False)
        """

        self._instance = None
        self.yVector = Vec2(-1, 0)
        self.cur_state = random.choice(self.STATES)
        self.cur_speed = 0

        self.actor = Actor(rand_texture['path'])

        self.actor.loadAnims(
            {'walk': AssetPaths.Pedestrian.PEDESTRIAN_MOTIONS['walk']})
        self.actor.loadAnims(
            {'run': AssetPaths.Pedestrian.PEDESTRIAN_MOTIONS['run']})
        self.actor.loadAnims(
            {'idle': AssetPaths.Pedestrian.PEDESTRIAN_MOTIONS['idle']})

        self.actor.setHpr(self.actor.getH() + 180,
                          self.actor.getP() + 0, self.actor.getR() + 0)
        self.actor.setPos(0, 0, -self.HEIGHT / 2)
        # self.origin.setPos(0, 0, 1)

        self.actor.loop(self.cur_state, fromFrame=10, toFrame=50)

        self.setup_collisions()

        if self.render:
            self._instance = self.actor.instanceTo(self.origin)
            self.show_coordinates()

    def setup_collisions(self):

        self.cTrav = CollisionTraverser()

        ground_ray = CollisionRay(0, 0, 2, 0, 0, -1)
        ground_col = CollisionNode('pedestrian_ground_ray')
        ground_col.addSolid(ground_ray)
        ground_col.setFromCollideMask(CollideMask.bit(0))
        ground_col.setIntoCollideMask(CollideMask.allOff())

        ground_col_np = self.origin.attachNewNode(ground_col)
        self.ground_handler = CollisionHandlerQueue()
        self.cTrav.addCollider(ground_col_np, self.ground_handler)

        # obstacle_ray = CollisionRay(0, 0, 0.5,  # origin pos
        #                                  1, 0, 0)  # dir
        # obstacle_col = CollisionNode("pedestrian_obstacle_ray")
        # obstacle_col.addSolid(obstacle_ray)
        # obstacle_col.setFromCollideMask(CollideMask.bit(0))

        # obstacle_col_np = self.origin.attachNewNode(obstacle_col)
        # self.obstacle_handler = CollisionHandlerQueue()
        # self.cTrav.addCollider(obstacle_col_np, self.obstacle_handler)

        ground_col_np.show()
        # obstacle_col_np.show()

    def set_heading_theta(self, heading_theta, in_rad=True) -> None:
        """
        Set heading theta for this object
        :param heading_theta: float
        :param in_rad: when set to True, heading theta should be in rad, otherwise, in degree
        """

        h = panda_heading(heading_theta)
        if in_rad:
            h = h * 180 / np.pi
        self.origin.setH(h)
        # hprInterval = self.origin.hprInterval(2, Point3(h, 0, 0), startHpr=Point3(self.origin.getH(), 0, 0))
        # hprInterval.start()

    def set_angular_velocity(self, angular_velocity, in_rad=True):
        if not in_rad:
            angular_velocity = angular_velocity / 180 * np.pi
        self._body.setAngularVelocity(LVector3(0, 0, angular_velocity))

    def set_velocity(self, direction=[1, 0], value=None, in_local_frame=False):
        """
        direction: use (1, 0) as default, dont care about the direction, change speed only 
        value: the speed
        inlocal_frame
        """

        self.set_roll(0)
        self.set_pitch(0)
        if in_local_frame:
            from metadrive.engine.engine_utils import get_engine
            engine = get_engine()
            direction = LVector3(*direction, 0.)
            direction[1] *= -1
            ret = engine.worldNP.getRelativeVector(self.origin, direction)
            direction = ret
        # self.prev_direction.angleRad(direction)
        # self.set_heading_theta()

        speed = (norm(direction[0], direction[1]) + 1e-6)
        if value is not None:
            norm_ratio = value / speed
        else:
            norm_ratio = 1

        # print("set_velocity", speed, speed * norm_ratio)
        self._body.setLinearVelocity(
            LVector3(direction[0] * norm_ratio, direction[1] * norm_ratio,
                     self._body.getLinearVelocity()[-1])
        )
        # angle = self.prev_direction.angleRad(direction)
        # print(angle)
        # self.set_model_heading_theta(angle, in_rad=True)

        self.cur_speed = value

        self.standup()
        self._instance = self.actor.instanceTo(self.origin)

        def set_actor_anim(target_anim, fromFrame=None, toFrame=None):
            # cur_state != target_state
            if not self.actor.get_anim_control(target_anim).isPlaying():
                self.actor.stop(self.cur_state)
                self.actor.loop(
                    target_anim, fromFrame=fromFrame, toFrame=toFrame)
                self.cur_state = target_anim

        if speed * norm_ratio >= 2:  # run
            set_actor_anim('run', fromFrame=10, toFrame=50)
        elif speed * norm_ratio < 0.01:  # idle
            set_actor_anim('idle')
        else:  # walk
            set_actor_anim('walk', fromFrame=10, toFrame=50)

    def move(self, orientation, distance):
        # if self.node().is_on_ground():
        # print(self.origin.get_pos(), type(self.origin.get_pos()))
        next_pos = self.origin.get_pos() + LPoint3f(*orientation, 0) * distance
        self.origin.set_pos(next_pos) # actor

    
    @property
    def LENGTH(self):
        return self.RADIUS

    @property
    def WIDTH(self):
        return self.RADIUS

    @property
    def top_down_width(self):
        return self.RADIUS

    @property
    def top_down_length(self):
        return self.RADIUS


class PedestrianNavigation(PedestrianBase):

    def __init__(self, position, heading_theta, random_seed=None, name=None, config=None, set_friction=None):

        super(PedestrianNavigation, self).__init__(
            position, heading_theta, random_seed, name, set_friction=set_friction)
        AssetLoader.init_loader(get_engine())
        self.lidar = self.engine.get_sensor("lidar")

        # self.set_position(position, self.HEIGHT)

        if config is not None:
            self.config.update(config, allow_add_new_key=True)

        # config = {'spawn_lane_index': ('-1X1_1_', '-1X1_0_', 0), 'spawn_longitude': 0, 'enable_reverse': False,
        #  'show_navi_mark': False, 'show_dest_mark': False, 'show_lidar': True, 'show_lane_line_detector': True,
        #  'show_side_detector': True}
        if self.config.get('need_navigation', False):
            self.spawn_place = (0, 0)
            self.add_navigation()  # default added
        # self.speed = 0

    def add_navigation(self):
        self.config.update(
            {"need_navigation": True, "destination": None}, allow_add_new_key=True)
        self.config.update(
            {"show_dest_mark": True, "show_navi_mark": True}, allow_add_new_key=True)
        self.config.update({"show_lane_line_detector": True, "show_lidar": True,
                           "show_side_detector": True}, allow_add_new_key=True)

        # if not self.config["need_navigation"]:
        #     return
        # navi = self.config["navigation_module"]
        # if navi is None:
        navi = PedestrianNodeNetworkNavigation
        self.navigation = navi(
            # self.engine,
            show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
            random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
            show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
            show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"],
            panda_color=self.panda_color,
            name=self.name,
            vehicle_config={"show_line_to_navi_mark": True}
        )
        self.navigation.reset(self)

    def _state_check(self, debug=False):
        """
        Check States and filter to update info
        """
        # print(self.engine.physics_world.dynamic_world.getRigidBodies())
        heightest_z = -1000
        result_1 = self.engine.physics_world.static_world.contactTest(
            self.origin.node(), True)
        result_2 = self.engine.physics_world.dynamic_world.contactTest(
            self.origin.node(), True)
        contacts = set()
        for contact in result_1.getContacts() + result_2.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            node = node0 if node1.getName() == MetaDriveType.PEDESTRIAN else node1
            if node.getName() == "detector_mask":
                continue
            name = node.getName()
            # node.origin.getZ()
            distance = contact.get_manifold_point().getDistance()
            new_z = self.origin.getZ() - distance
            heightest_z = max(new_z, heightest_z)
            if debug:
                contacts.add((name, self.origin.getZ() - distance, distance))
            else:
                contacts.add(name)
        if heightest_z != -1000:
            self.origin.setZ(self.origin.getZ() - distance)
        return contacts

    @staticmethod
    def _preprocess_action(action):
        if action is None:
            return None, {"raw_action": None}
        action = safe_clip_for_small_array(action, -1, 1)
        return action, {'raw_action': (action[0], action[1])}

    def _set_action(self, action):
        if action is None:
            return
        # base_speed = 1
        steering = action[0]
        acceleration = action[1]

        # self.cur_speed = self.cur_speed + self.cur_speed * (1 + acceleration)
        speed = max(0, self.cur_speed + acceleration)
        heading = self.origin.getH() + steering
        # self.set_heading_theta(steering * 5, in_rad=False)
        # self.set_heading_theta(self.heading_theta + steering, in_rad=True)
        self.set_heading_theta(heading, in_rad=False)
        self.set_velocity(value=speed, in_local_frame=True)

        # steering = action[0]
        # self.throttle_brake = action[1]
        # self.steering = steering
        # self.system.setSteeringValue(self.steering * self.max_steering, 0)
        # self.system.setSteeringValue(self.steering * self.max_steering, 1)
        # self._apply_throttle_brake(action[1])

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        action, step_info = self._preprocess_action(action)

        # self.last_position = self.position  # 2D vector
        # self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar
        self._set_action(action)
        return step_info

    @property
    def reference_lanes(self):
        return self.navigation.current_ref_lanes

    @property
    def lane(self):
        return self.navigation.current_lane

    @property
    def lane_index(self):
        return self.navigation.current_lane.index


    def follow(self, obj2):
        vec_to = self.origin.getPos() - obj2.origin.getPos()
        vec_to_2d = vec_to.getXy()
        dist_to = vec_to_2d.length()
        vec_to_2d.normalize()

        heading = self.yVector.signedAngleDeg(vec_to_2d)  # degree between two peds

        speed = (dist_to - 2) / 2  # catch up in 2 sec
        if dist_to > 2:
            self.set_heading_theta(heading, in_rad=False)
            self.set_velocity(value=speed, in_local_frame=True)
        else:
            self.set_velocity(value=0, in_local_frame=True)


    def set_on_ground(self):
        engine = get_engine()
        self.standup()
        # self.ralphGroundHandler.entries
        self.cTrav.traverse(engine.render)

        if self.ground_handler.getNumEntries() == 0:
            # print("set_on_ground", list(self.ground_handler.entries))
            return
        entries = list(self.ground_handler.entries)
        entries.sort(key=lambda x: x.getSurfacePoint(engine.render).getZ())
        entry = self.ground_handler.getEntry(0)
        new_z = entry.getSurfacePoint(engine.render).getZ()

        print("set_on_ground", len(entries), new_z)
        self.origin.setZ(new_z)


    def get_obj_infos(self, vehicle, objs):
        res = []
        # ego_position = vehicle.origin.getPos()
        ego_position = vehicle.position
        for obj in objs:
            # obj_position = obj.origin.getPos()

            # assert isinstance(vehicle, IDMVehicle or Base), "Now MetaDrive Doesn't support other vehicle type"
            vec_to_2d = ego_position - obj.position  
            dist_to = vec_to_2d.length()
            vec_to_2d_norm = vec_to_2d.normalize()

            heading = vehicle.yVector.signedAngleDeg(vec_to_2d_norm)

            degree = vehicle.origin.getH() - heading
            res.append(degree)
            res.append(dist_to)

            relative_position = obj.convert_to_local_coordinates(obj.position, ego_position)
            # It is possible that the centroid of other vehicle is too far away from ego but lidar shed on it.
            # So the distance may greater than perceive distance.
            res.append(relative_position[0])
            res.append(relative_position[1])

            relative_velocity = obj.convert_to_local_coordinates(
                obj.velocity, vehicle.velocity
            )
            res.append(relative_velocity[0] )
            res.append(relative_velocity[1] )
            # [heading, distance, rel_pos_x, rel_pos_y, rel_velocity_x, rel_velocity_y]
        return res
    

    def get_posible_hit_obstacles(self, left_time_in_sec=3, range_in_degree=30):
        self.lidar.perceive(base_vehicle=self, physics_world=self.engine.physics_world.static_world, num_lasers=2, distance=8, show=True)
        objs = self.lidar.get_surrounding_objects(self)
        res = self.get_obj_infos(self, objs)

        possible_obstacles = []
        for idx, obj in enumerate(objs):
            degree, distance, rel_pos_x, rel_pos_y, rel_velocity_x, rel_velocity_y = res[idx*6: (idx+1)*6]

            speed = np.sqrt(rel_velocity_x * rel_velocity_x +
                            rel_velocity_y * rel_velocity_y)

            time_to_hit_in_sec = distance / (speed + 1e-6)

            if time_to_hit_in_sec < left_time_in_sec and abs(degree) < range_in_degree:
                possible_obstacles.append(
                    (obj, degree, distance, time_to_hit_in_sec))

        return possible_obstacles


    def do_obstacle_aviodance(self):
        obstacles = sorted(self.get_posible_hit_obstacles(), key=lambda x: x[-1])
        if len(obstacles) == 0:
            return

        # obj, degree, distance, time_to_hit_in_sec = obstacles[0]
        # print("do_obstacle_aviodance", obj, degree, distance, time_to_hit_in_sec)
        # print("new heading", self.origin.getH() + 90)
        self.set_heading_theta(self.origin.getH() + 90, in_rad=False)
        self.set_velocity(value=self.cur_speed, in_local_frame=True)


    def get_current_ground(self):
        return self.ground_detector.perceive(self, get_engine().physics_world.dynamic_world)
