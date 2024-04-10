import copy
import logging
from collections import namedtuple
from typing import Dict

import os
import math
import matplotlib.pyplot as plt
import numpy as np
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.examples.ppo_expert.custom_expert import get_dest_heading
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.orca_planning_utils import OrcaPlanning
from metadrive.utils import merge_dicts


BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    # Traffic vehicles will be respawned, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"


class PGTrafficManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(PGTrafficManager, self).__init__()

        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        else:
            raise ValueError("No such mode named {}".format(self.mode))

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        if self.mode != TrafficMode.Respawn:
            for v in engine.agent_manager.active_agents.values():
                ego_lane_idx = v.lane_index[:-1]
                ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                if len(self.block_triggered_vehicles) > 0 and \
                        ego_road == self.block_triggered_vehicles[-1].trigger_road:
                    block_vehicles = self.block_triggered_vehicles.pop()
                    self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values())
        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()

    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                if self.mode == TrafficMode.Trigger:
                    v_to_remove.append(v)
                elif self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                    v_to_remove.append(v)
                else:
                    raise ValueError("Traffic mode error: {}".format(self.mode))
        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
            if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed())
                self._traffic_vehicles.append(new_v)

        return dict()

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(PGTrafficManager, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self.block_triggered_vehicles = []
        self._traffic_vehicles = []

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        if self.mode == TrafficMode.Respawn:
            return len(self._traffic_vehicles)
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self) -> Dict:
        """
        Return all traffic vehicles' states
        :return: States of all vehicles
        """
        states = dict()
        traffic_states = dict()
        for vehicle in self._traffic_vehicles:
            traffic_states[vehicle.index] = vehicle.get_state()

        # collect other vehicles
        if self.mode != TrafficMode.Respawn:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    traffic_states[vehicle.index] = vehicle.get_state()
        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v.get_state()
                                      for k, v in pending_obj.items()}, allow_new_keys=True
        )
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v_count[0].get_state()
                                      for k, v_count in dying_obj.items()},
            allow_new_keys=True
        )

        states[OBJECT_TO_AGENT] = copy.deepcopy(self.engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(self.engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        Special handling for first states of traffic vehicles
        :return: States of all vehicles
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        # collect other vehicles
        if self.mode != TrafficMode.Respawn:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    init_state["enable_respawn"] = vehicle.enable_respawn
                    vehicles[vehicle.index] = init_state
        return vehicles

    def _propose_vehicle_configs(self, lane: AbstractLane):
        potential_vehicle_configs = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        # Only choose given number of vehicles
        for long in vehicle_longs:
            random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs

    def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float):
        total_num = len(self.respawn_lanes)
        for lane in self.respawn_lanes:
            _traffic_vehicles = []
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
            self.np_random.shuffle(vehicle_longs)
            for long in vehicle_longs[:int(np.ceil(traffic_density * len(vehicle_longs)))]:
                # if self.np_random.rand() > traffic_density and abs(lane.length - InRampOnStraight.RAMP_LEN) > 0.1:
                #     # Do special handling for ramp, and there must be vehicles created there
                #     continue
                vehicle_type = self.random_vehicle_type()
                traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long}
                traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                self._traffic_vehicles.append(random_v)

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

            from metadrive.policy.idm_policy import IDMPolicy
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def _get_available_respawn_lanes(self, map: BaseMap) -> list:
        """
        Used to find some respawn lanes
        :param map: select spawn lanes from this map
        :return: respawn_lanes
        """
        respawn_lanes = []
        respawn_roads = []
        for block in map.blocks:
            roads = block.get_respawn_roads()
            for road in roads:
                if road in respawn_roads:
                    respawn_roads.remove(road)
                else:
                    respawn_roads.append(road)
        for road in respawn_roads:
            respawn_lanes += road.get_lanes(map.road_network)
        return respawn_lanes

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self._traffic_vehicles = None
        self.block_triggered_vehicles = None

        # traffic property
        self.mode = None
        self.random_traffic = None
        self.density = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self._traffic_vehicles.__repr__()

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        if not self.random_traffic:
            super(PGTrafficManager, self).seed(random_seed)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    def get_state(self):
        ret = super(PGTrafficManager, self).get_state()
        ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
        flat = []
        for b_v in self.block_triggered_vehicles:
            flat.append((b_v.trigger_road.start_node, b_v.trigger_road.end_node, b_v.vehicles))
        ret["block_triggered_vehicles"] = flat
        return ret

    def set_state(self, state: dict, old_name_to_current=None):
        super(PGTrafficManager, self).set_state(state, old_name_to_current)
        self._traffic_vehicles = list(
            self.get_objects([old_name_to_current[name] for name in state["_traffic_vehicles"]]).values()
        )
        self.block_triggered_vehicles = [
            BlockVehicles(trigger_road=Road(s, e), vehicles=[old_name_to_current[name] for name in v])
            for s, e, v in state["block_triggered_vehicles"]
        ]


# For compatibility check
TrafficManager = PGTrafficManager


class MixedPGTrafficManager(PGTrafficManager):
    def _create_respawn_vehicles(self, *args, **kwargs):
        raise NotImplementedError()

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]

            from metadrive.policy.idm_policy import IDMPolicy
            from metadrive.policy.expert_policy import ExpertPolicy
            # print("===== We are initializing {} vehicles =====".format(len(selected)))
            # print("Current seed: ", self.engine.global_random_seed)
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                if self.np_random.random() < self.engine.global_config["rl_agent_ratio"]:
                    # print("Vehicle {} is assigned with RL policy!".format(random_v.id))
                    self.add_policy(random_v.id, ExpertPolicy, random_v, self.generate_seed())
                else:
                    self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

import cv2

class HumanoidManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(HumanoidManager, self).__init__()

        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        # self.walkable_mask, self.walkable_offset_x, self.walkable_offset_y = self.get_walkable_mask(map)
        self.walkable_mask, self.walkable_offset_x, self.walkable_offset_y = self.get_walkable_mask_new(map)

        self.num_humanoid_agent = 1 #20
        self.planning = OrcaPlanning() # "./orca_algo/task_examples_demo/custom_road_template.xml"
        self.planning.generate_template_xml(self.walkable_mask)

        self.starts, self.goals = self.planning.random_starts_and_goals(self.walkable_mask[..., 0], self.num_humanoid_agent)
        if self.num_humanoid_agent > 0:
            self.planning.get_planning(self.starts, self.goals)
        
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        # self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_humanoid_once(map, traffic_density)
            # self._create_vehicles_once(map, traffic_density)
            
        else:
            raise ValueError("No such mode named {}".format(self.mode))

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        if self.mode != TrafficMode.Respawn:
            for v in engine.agent_manager.active_agents.values():
                ego_lane_idx = v.lane_index[:-1]
                ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                if len(self.block_triggered_vehicles) > 0 and \
                        ego_road == self.block_triggered_vehicles[-1].trigger_road:
                    block_vehicles = self.block_triggered_vehicles.pop()
                    self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values())
        for v in self._traffic_vehicles:
            try:
                p = self.engine.get_policy(v.name)
                v.before_step(p.act())
            except Exception:
                pass
        if self.num_humanoid_agent > 0: 
            self.step_action()
        return dict()

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

            from metadrive.policy.idm_policy import IDMPolicy
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def _create_humanoid_once(self, map: BaseMap, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        
        potential_vehicle_configs = []
        for sidewalk_index in map.sidewalks.values():
            potential_vehicle_configs += self._propose_humanoid_configs(sidewalk_index['polygon'])
        
        total_humanoids = len(self.starts)
        self.humanoid_on_block = []
        self.np_random.shuffle(potential_vehicle_configs)
        selected = potential_vehicle_configs[:min(total_humanoids, len(potential_vehicle_configs))]
        # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

        for v_config in selected:
            humanoid_type = self.random_humanoid_type()
            v_config.update(self.engine.global_config["traffic_vehicle_config"])
            random_v = self.spawn_object(humanoid_type, vehicle_config=v_config)
            self.humanoid_on_block.append(random_v.name)

    def draw_objects(self, objects, canvas):
        w = object.WIDTH
        h = object.LENGTH
        position = [object.position[0], object.position[1]]
        # As the following rotate code is for left-handed coordinates,
        # so we plus -1 before the heading to adapt it to right-handed coordinates
        heading = objects.heading_theta
        heading = heading if abs(heading) > 2 * np.pi / 180 else 0
        angle = -np.rad2deg(heading)
        box = [p for p in [(-h / 2, -w / 2), (-h / 2, w / 2), (h / 2, w / 2), (h / 2, -w / 2)]]
        box_rotate = [p.rotate(angle) + position for p in box]

        pts = box_rotate
        pts = pts.reshape((-1,1,2))
        canvas = cv2.fillPoly(canvas, [pts], [255, 255, 255])
        return canvas

    def get_walkable_mask(self, map, objects=None):
        line_sample_interval = 2
        all_lanes = map.get_map_features(line_sample_interval)
        crosswalk_keys = list(filter(lambda x: "CRS_" in x, all_lanes.keys()))
        sidewalk_keys = list(filter(lambda x: "SDW_" in x, all_lanes.keys()))
        all_pts = []

        img = np.zeros((256, 256, 3), np.uint8)
        for key in crosswalk_keys:
            obj = all_lanes[key]
            # pts = np.array([(p[0] - 100, p[1]) for p in np.array(obj["polygon"]) * 2 + 100], np.int32)
            pts = np.array([(p[0], p[1]) for p in np.array(obj["polygon"]) + 50], np.int32)
            # pts = np.array([(p[0], p[1]) for p in np.array(obj["polygon"])], np.int32)
            all_pts.append(pts)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img, [pts], [255, 255, 255])
            plt.fill(pts[:, 0, 0], pts[:, 0, 1])

        for key in sidewalk_keys:
            obj = all_lanes[key]
            # pts = np.array([(p[0] - 100, p[1]) for p in np.array(obj["polygon"]) * 2 + 100], np.int32)
            pts = np.array([(p[0], p[1]) for p in np.array(obj["polygon"]) + 50], np.int32)
            # pts = np.array([(p[0], p[1]) for p in np.array(obj["polygon"])], np.int32)

            all_pts.append(pts)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img, [pts], [255, 255, 255])
            plt.fill(pts[:, 0, 0], pts[:, 0, 1])
        offset_x, offset_y = 0, 0 
        cfg = self.engine.global_config['map']
        save_mask_name = f"output/map_masks/test_walkable_mask_old_{cfg}.png"
        if not os.path.exists(save_mask_name):
            cv2.imwrite(save_mask_name, img)
        return img, offset_x, offset_y


    def get_walkable_mask_new(self, map, objects=None):
        def get_mask_range(all_pts):
            max_w = 0; max_h = 0
            min_x = 0; min_y = 0
            
            for pt in all_pts:
                max_w = max(max_w, pt[:, 0].max())
                max_h = max(max_h, pt[:, 1].max())

                min_x = min(min_x, pt[:, 0].min())
                min_y = min(min_y, pt[:, 1].min())

            w = max_w - min_x
            h = max_h - min_y
            return w, h, min_x, min_y
        
        line_sample_interval = 1
        all_lanes = map.get_map_features(line_sample_interval)
        # print(all_lanes.keys())  #"CRS_('1X0_0_', '1X0_1_', 0)": {'type': 'ROAD_EDGE_SIDEWALK', 'polygon': [[69.45, -42.0], [69.45, -45.0], [69.45, -48.0], [59.65, -48.0], [59.65, -45.0], [59.65, -42.0]], 'height': None},
        crosswalk_keys = list(filter(lambda x: "CRS_" in x, all_lanes.keys()))
        sidewalk_keys = list(filter(lambda x: "SDW_" in x, all_lanes.keys()))

        # print('traffic_manager, crosswalk & sidewalk keys: \,', crosswalk_keys[0].keys(), '\n', sidewalk_keys[0].keys())
        # assert False
        all_pts = []

        for key in crosswalk_keys:
            obj = all_lanes[key]
            pts = np.array([(p[0], p[1]) for p in np.array(obj["polygon"])], np.int32)
            all_pts.append(pts)

        for key in sidewalk_keys:
            obj = all_lanes[key]
            pts = np.array([(p[0], p[1]) for p in np.array(obj["polygon"])], np.int32)
            all_pts.append(pts)

        map_x, map_y, offset_x, offset_y = get_mask_range(all_pts)

        img = np.zeros((map_y, map_x, 3), np.uint8)

        for pts in all_pts:
            pts[:, 0] -= offset_x
            pts[:, 1] -= offset_y
            cv2.fillPoly(img, [pts], [255, 255, 255])
        img = cv2.flip(img, 0) 

        cfg = self.engine.global_config['map']
        save_mask_name = f"output/map_masks/test_walkable_mask_{cfg}.png"
        if not os.path.exists(save_mask_name):
            cv2.imwrite(save_mask_name, img)
        # exit(0)
        # plt.axis('equal')
       
        return img, offset_x, offset_y
    

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    def _propose_vehicle_configs(self, lane: AbstractLane):
        potential_vehicle_configs = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        # Only choose given number of vehicles
        for long in vehicle_longs:
            random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs
    
    def _propose_humanoid_configs(self, polygon):
        def get_random_points_in_polygon(polygon, number_point):
            points = polygon[:number_point]
            return points

        potential_vehicle_configs = []
        points = get_random_points_in_polygon(polygon, 3)
        heading = 0

        for point in points:
            random_vehicle_config = {"spawn_position_heading": [point, heading]}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs

    def random_humanoid_type(self):
        from metadrive.component.agents.pedestrian.pedestrian_type import SimplePedestrian
        
        return SimplePedestrian

    def step_action(self):
        def apply_actions(objs, dest_pos, speed):
            # print("------------------------------------------")
            for objname, pos, speed in zip(objs, dest_pos, speed):
                obj = list(self.engine.get_object(objname).values())[0]
                pos = pos[0] + self.walkable_offset_x, pos[1] + self.walkable_offset_y

                # pos = self.planning.coord_orca_to_md(pos)
                heading = get_dest_heading(obj, pos)
                # print(heading)
                # obj.set_heading(heading)
                speed = speed / self.engine.global_config["physics_world_step_size"]
                obj.set_anim_by_speed(speed)



                obj.set_position(pos)
                obj._body.setAngularMovement(heading)
                # obj.set_roll(heading)
                # print(heading)
                # obj._body.setLinearMovement(LPoint3f(0 , 1, 0) * 3, True)
            # print("------------------------------------------")

        objs = self.get_objects(self.humanoid_on_block)

        if not self.planning.has_next():
            self.starts = self.goals
            _, self.goals = self.planning.random_starts_and_goals(self.walkable_mask[..., 0], self.num_humanoid_agent)
            self.planning.get_planning(self.starts, self.goals)

        # dest_pos = self.planning.get_next()
        dest_pos, speed = self.planning.get_next(return_speed=True)
        
        apply_actions(objs, dest_pos, speed)

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type
