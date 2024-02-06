
from metadrive.component.agents.pedestrian.base_pedestrian import BasePedestrian
from metadrive.component.pg_space import ParameterSpace, VehicleParameterSpace


class SimplePedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)
    # LENGTH = 1.
    # WIDTH = 1.85
    # HEIGHT = 1.37
    RADIUS = 0.35

    # REAR_WHEELBASE = 1.203
    # FRONT_WHEELBASE = 1.285
    # LATERAL_TIRE_TO_CENTER = 0.803
    # TIRE_WIDTH = 0.3
    MASS = 80
    # LIGHT_POSITION = (-0.67, 1.86, 0.22)

    # path = ['130/vehicle.gltf', (1, 1, 1), (0, -0.05, 0.1), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 1.  # meters

    @property
    def HEIGHT(self):
        return 1.37  # meters

    @property
    def WIDTH(self):
        return 1.  # meters