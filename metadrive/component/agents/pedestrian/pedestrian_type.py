
from metadrive.component.agents.pedestrian.base_pedestrian import BasePedestrian
from metadrive.component.pg_space import ParameterSpace, VehicleParameterSpace
from metadrive.constants import AssetPaths
from metadrive.utils.config import Config


class SimplePedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)
    
    RADIUS = 0.35
    MASS = 80

    # def __init__(self, vehicle_config: dict | Config = None, name: str = None, random_seed=None, position=None, heading=None, _calling_reset=True):
    #     super().__init__(vehicle_config, name, random_seed, position, heading, _calling_reset)

    #     self.random_actor = AssetPaths.Pedestrian.get_random_actor()

    @property
    def LENGTH(self):
        return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()
        return self.random_actor['height']


    @property
    def WIDTH(self):
        return 1.  # meters
    
    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()
        return self.random_actor['actor_path']
    
    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()
        return self.random_actor['motion_path']
