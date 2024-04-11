import math
import numpy as np
from metadrive.type import MetaDriveType
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pgblock.create_pg_block_utils import CreateAdverseRoad, CreateRoadFrom, create_bend_straight
from metadrive.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metadrive.component.road_network import Road
from metadrive.constants import PGLineType
from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace
from metadrive.constants import PGDrivableAreaProperty


class Roundabout(PGBlock):
    """
    roundabout class, the example is the same as Intersection
    """
    ID = "O"
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.ROUNDABOUT)
    SOCKET_NUM = 3
    RADIUS_IN = 20
    ANGLE = 60
    EXIT_PART_LENGTH = 35

    def __init__(self, *args, **kwargs):
        super(Roundabout, self).__init__(*args, **kwargs)
        self.intermediate_spawn_places = []

    def _try_plug_into_previous_block(self) -> bool:
        self.intermediate_spawn_places = []
        para = self.get_config(copy=False)
        no_cross = True
        attach_road = self.pre_block_socket.positive_road
        for i in range(4):
            exit_road, success = self._create_circular_part(
                attach_road, i, para[Parameter.radius_exit], para[Parameter.radius_inner], para[Parameter.angle]
            )
            no_cross = no_cross and success
            if i < 3:
                no_cross = CreateAdverseRoad(
                    exit_road,
                    self.block_network,
                    self._global_network,
                    ignore_intersection_checking=self.ignore_intersection_checking
                ) and no_cross
                attach_road = -exit_road
        # assert False
        self.add_respawn_roads([socket.negative_road for socket in self.get_socket_list()])
        return no_cross

    def _create_circular_part(self, road: Road, part_idx: int, radius_exit: float, radius_inner: float,
                              angle: float) -> (str, str, StraightLane, bool):
        """
        Create a part of roundabout according to a straight road
        """
        ### lane index: str(block_idx) + cls.ID + str(part_idx) + cls.DASH + str(road_idx) + cls.DASH  # e.g. ('1O0_0_', '1O0_1_', 0)
        none_cross = True
        self.set_part_idx(part_idx)
        radius_big = (self.positive_lane_num * 2 - 1) * self.lane_width + radius_inner

        # circular part 0
        segment_start_node = road.end_node  # previous road
        segment_end_node = self.add_road_node()
        segment_road = Road(segment_start_node, segment_end_node)
        # print(f'0: segment_road: {segment_road.start_node} - {segment_road.end_node}')  #>>> - 1O0_0_

        lanes = road.get_lanes(self._global_network) if part_idx == 0 else road.get_lanes(self.block_network)
        right_lane = lanes[-1]
        bend, straight = create_bend_straight(
            right_lane, 10, radius_exit, np.deg2rad(angle), True, self.lane_width, (PGLineType.BROKEN, PGLineType.SIDE)
        )
        # print(f'part{part_idx}_circular0_bend={bend.polygon}') 
        # print(f'part{part_idx}_circular0_straight={straight.polygon.tolist()}')

        ignore_last_2_part_start = self.road_node((part_idx + 3) % 4, 0)
        ignore_last_2_part_end = self.road_node((part_idx + 3) % 4, 0)
        none_cross = CreateRoadFrom(
            bend,
            self.positive_lane_num,
            segment_road,
            self.block_network,
            self._global_network,
            ignore_start=ignore_last_2_part_start,
            ignore_end=ignore_last_2_part_end,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and none_cross
        # print(f'1: segment_road: {segment_road.start_node} - {segment_road.end_node}') #>>> - 1O0_0_

        # set circular part 0 visualization    #####??
        for k, lane in enumerate(segment_road.get_lanes(self.block_network)):
            if k == self.positive_lane_num - 1:  # k=2
                lane.line_types = [PGLineType.NONE, PGLineType.SIDE]
            else: # k=0,1
                lane.line_types = [PGLineType.NONE, PGLineType.NONE]

        # circular part 1
        tool_lane_start = straight.position(-5, 0)
        tool_lane_end = straight.position(0, 0)
        tool_lane = StraightLane(tool_lane_start, tool_lane_end)

        bend, straight_to_next_iter_part = create_bend_straight(
            tool_lane, 10, radius_big, np.deg2rad(2 * angle - 90), False, self.lane_width,
            (PGLineType.BROKEN, PGLineType.SIDE)
        )

        # print(f'part{part_idx}_circular1_bend={bend.polygon}\n') 
        # print(f'part{part_idx}_circular1_straight={straight.polygon.tolist()}\n')
        segment_start_node = segment_end_node
        segment_end_node = self.add_road_node()
        segment_road = Road(segment_start_node, segment_end_node)
        # print(f'2: segment_road: {segment_road.start_node} - {segment_road.end_node}')  #1O0_0_ - 1O0_1_

        none_cross = CreateRoadFrom(
            bend,
            self.positive_lane_num,
            segment_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and none_cross
        # print(f'3: segment_road: {segment_road.start_node} - {segment_road.end_node}') #1O0_0_ - 1O0_1_

        self.intermediate_spawn_places.append(segment_road.get_lanes(self.block_network))

        # circular part 2 and exit straight part
        length = self.EXIT_PART_LENGTH
        tool_lane_start = straight_to_next_iter_part.position(-5, 0)
        tool_lane_end = straight_to_next_iter_part.position(0, 0)
        tool_lane = StraightLane(tool_lane_start, tool_lane_end)

        bend, straight = create_bend_straight(
            tool_lane, length, radius_exit, np.deg2rad(angle), True, self.lane_width,
            (PGLineType.BROKEN, PGLineType.SIDE)
        )
        # print(f'part{part_idx}_circular2_bend={bend.polygon}\n') #, bend start & end: {bend.start}.. {bend.end}\n')
        # print(f'part{part_idx}_circular2_straight={straight.polygon.tolist()}\n')#, straight start & end: {straight.start}.. {straight.end}\n')
        segment_start_node = segment_end_node
        segment_end_node = self.add_road_node() if part_idx < 3 else self.pre_block_socket.negative_road.start_node
        segment_road = Road(segment_start_node, segment_end_node)
        # print(f'4: segment_road: {segment_road.start_node} - {segment_road.end_node}') #1O0_1_ - 1O0_2_

        none_cross = CreateRoadFrom(
            bend,
            self.positive_lane_num,
            segment_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and none_cross
        # print(f'5: segment_road: {segment_road.start_node} - {segment_road.end_node}') #1O0_1_ - 1O0_2_

        # set circular part 2 (curve) visualization
        for k, lane in enumerate(segment_road.get_lanes(self.block_network)):
            if k == self.positive_lane_num - 1:
                lane.line_types = [PGLineType.NONE, PGLineType.SIDE]
            else:
                lane.line_types = [PGLineType.NONE, PGLineType.NONE]

        exit_start = segment_end_node
        exit_end = self.add_road_node()
        segment_road = Road(exit_start, exit_end)
        # print(f'6: segment_road: {segment_road.start_node} - {segment_road.end_node}') #1O0_2_ - 1O0_3_

        if part_idx < 3:
            none_cross = CreateRoadFrom(
                straight,
                self.positive_lane_num,
                segment_road,
                self.block_network,
                self._global_network,
                ignore_intersection_checking=self.ignore_intersection_checking
            ) and none_cross
            self.add_sockets(self.create_socket_from_positive_road(segment_road))
            # print(f'7: segment_road: {segment_road.start_node} - {segment_road.end_node}') #1O0_2_ - 1O0_3_

        #  add circular part 3 at last
        segment_start = self.road_node(part_idx, 1)
        segment_end = self.road_node((part_idx + 1) % 4, 0)
        segment_road = Road(segment_start, segment_end)
        # print(f'8: segment_road: {segment_road.start_node} - {segment_road.end_node}') #1O0_1_ - 1O1_0_

        tool_lane_start = straight_to_next_iter_part.position(-6, 0)
        tool_lane_end = straight_to_next_iter_part.position(0, 0)
        tool_lane = StraightLane(tool_lane_start, tool_lane_end)

        beneath = (self.positive_lane_num * 2 - 1) * self.lane_width / 2 + radius_exit
        cos = math.cos(np.deg2rad(angle))
        radius_this_seg = beneath / cos - radius_exit

        bend, _ = create_bend_straight(
            tool_lane, 5, radius_this_seg, np.deg2rad(180 - 2 * angle), False, self.lane_width,
            (PGLineType.BROKEN, PGLineType.SIDE)
        )
        # print(f'part{part_idx}_circular3_bend={bend.polygon}\n') 
        CreateRoadFrom(
            bend,
            self.positive_lane_num,
            segment_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        # print(f'9: segment_road: {segment_road.start_node} - {segment_road.end_node}') #1O0_1_ - 1O1_0_

        # set circular part 2 visualization
        for k, lane in enumerate(segment_road.get_lanes(self.block_network)):
            if k == 0:
                if self.positive_lane_num > 1:
                    lane.line_types = [PGLineType.CONTINUOUS, PGLineType.BROKEN]
                else:
                    lane.line_types = [PGLineType.CONTINUOUS, PGLineType.NONE]
            else:
                lane.line_types = [PGLineType.BROKEN, PGLineType.BROKEN]

        return Road(exit_start, exit_end), none_cross

    def get_socket(self, index: int) -> PGBlockSocket:
        socket = super(Roundabout, self).get_socket(index)
        if socket.negative_road in self.get_respawn_roads():
            self._respawn_roads.remove(socket.negative_road)
        return socket

    def get_intermediate_spawn_lanes(self):
        """Filter out half of the vehicles."""
        return self.get_respawn_lanes() + self.intermediate_spawn_places


### sidewalk
    # def _generate_sidewalk_from_line

    def _generate_crosswalk_from_line(self, lane, sidewalk_height=None, lateral_direction=1):
        """
        Construct the sidewalk for this lane
        Args:
            block:

        Returns:
        """
        # print(lane.polygon)
        # print(lane.start_lat)
        # assert False
        # print('roudabout lane index: ' ,  lane.index) # ('1O0_0_', '1O0_1_', 0)

        if '_3_' in lane.index[1] or '_3_' in lane.index[0]:
            crosswalk_width = lane.width * 3  ## length
            start_lat = +lane.width_at(0) - crosswalk_width / 2 - 0.7
            side_lat = start_lat + crosswalk_width - 0.7
        else:
            return
            crosswalk_width = lane.width * 1.9
            start_lat = +lane.width_at(0) - crosswalk_width / 2 -2
            side_lat = start_lat + crosswalk_width -2

        build_at_start = True
        build_at_end = True
        if build_at_end:
            longs = np.array([lane.length - PGDrivableAreaProperty.SIDEWALK_LENGTH, lane.length, lane.length + PGDrivableAreaProperty.SIDEWALK_LENGTH])
            key = "CRS_" + str(lane.index)
            self.build_crosswalk_block(key, lane, sidewalk_height, lateral_direction, longs, start_lat, side_lat)
            
        if build_at_start:
            longs = np.array([0 - PGDrivableAreaProperty.SIDEWALK_LENGTH, 0, 0 + PGDrivableAreaProperty.SIDEWALK_LENGTH])
            key = "CRS_" + str(lane.index) + "_S"
            self.build_crosswalk_block(key, lane, sidewalk_height, lateral_direction, longs, start_lat, side_lat)
