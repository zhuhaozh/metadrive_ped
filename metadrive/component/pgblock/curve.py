import numpy as np

from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace
from metadrive.component.pgblock.create_pg_block_utils import CreateAdverseRoad, CreateRoadFrom, create_bend_straight
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.road_network import Road
from metadrive.constants import PGLineType
from metadrive.constants import PGDrivableAreaProperty


class Curve(PGBlock):
    """
        2 - - - - - - - - - -
       / 3 - - - - - - - - - -
      / /
     / /
    0 1
    """
    ID = "C"
    SOCKET_NUM = 1
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.CURVE)

    def _try_plug_into_previous_block(self) -> bool:
        parameters = self.get_config()
        basic_lane = self.positive_basic_lane
        lane_num = self.positive_lane_num

        # part 1
        start_node = self.pre_block_socket.positive_road.end_node  # >>>
        end_node = self.add_road_node()  #1C0_0_
        positive_road = Road(start_node, end_node)
        length = parameters[Parameter.length]
        direction = parameters[Parameter.dir]
        angle = parameters[Parameter.angle]
        radius = parameters[Parameter.radius]
        curve, straight = create_bend_straight(
            basic_lane,
            length,
            radius,
            np.deg2rad(angle),
            direction,
            width=basic_lane.width,
            line_types=(PGLineType.BROKEN, self.side_lane_line_type)
        )
        no_cross = CreateRoadFrom(
            curve,
            lane_num,
            positive_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking,
            side_lane_line_type=self.side_lane_line_type,
            center_line_type=self.center_line_type
        )
        if not self.remove_negative_lanes:
            no_cross = CreateAdverseRoad(
                positive_road,
                self.block_network,
                self._global_network,
                ignore_intersection_checking=self.ignore_intersection_checking,
                side_lane_line_type=self.side_lane_line_type,
                center_line_type=self.center_line_type
            ) and no_cross

        # part 2
        start_node = end_node  #1C0_0_
        end_node = self.add_road_node() #1C0_1_
        positive_road = Road(start_node, end_node)
        no_cross = CreateRoadFrom(
            straight,
            lane_num,
            positive_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking,
            side_lane_line_type=self.side_lane_line_type,
            center_line_type=self.center_line_type
        ) and no_cross
        if not self.remove_negative_lanes:
            no_cross = CreateAdverseRoad(
                positive_road,
                self.block_network,
                self._global_network,
                ignore_intersection_checking=self.ignore_intersection_checking,
                side_lane_line_type=self.side_lane_line_type,
                center_line_type=self.center_line_type
            ) and no_cross

        # common properties
        self.add_sockets(self.create_socket_from_positive_road(positive_road))
        return no_cross

    def _generate_crosswalk_from_line(self, lane, sidewalk_height=None, lateral_direction=1):
        ### TODO1: polygon
        """
        Construct the sidewalk for this lane
        Args:
            block:

        Returns:
        """
        crosswalk_width = lane.width * 3
        start_lat = +lane.width_at(0) - crosswalk_width / 2 - 0.7
        side_lat = start_lat + crosswalk_width - 0.7
        # # if ('1C0_0_', '1C0_1_', 0) == lane.index or ('1C0_0_', '1C0_1_', 1) == lane.index:
        # if '0_0_' in lane.index[0]  and '0_1_' in lane.index[1]:
        #     build_at_start = False
        #     build_at_end = True
        # elif ('0_1_' in lane.index[0] and '0_0_' in lane.index[1]):
        #     build_at_start = True
        #     build_at_end = False
        # else: return
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
