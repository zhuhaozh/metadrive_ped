import xml.etree.ElementTree as ET
import sys

sys.path.insert(0, './orca_algo/build')
import bind
import numpy as np

class OrcaPlanning:
    def __init__(self, template_xml_file, map_mask=None):
        self.template_xml_file = template_xml_file
        self.output_xml_file = 'output.xml'
        self.valid = False
        self.num_agent = -1
        self.next_positions = []
        self.prev_start_positions, self.prev_goals = None, None

    def set_agents(self, start_positions, goals):
        self.prev_start_positions, self.prev_goals = start_positions, goals

        tree = ET.parse(self.template_xml_file)
        root = tree.getroot()
        agents = root.findall('./agents')[0]
        agents.set("number", f"{len(start_positions)}")
        self.num_agent = len(start_positions)
        # """
        # movespeed element not found in XML file (or it is incorrect) at agent 13
        # agentsmaxnum element not found in XML file (or it is incorrect) at agent 13
        # timeboundary element not found in XML file (or it is incorrect) at agent 13
        # sightradius element not found in XML file (or it is incorrect) at agent 13
        # timeboundaryobst element not found in XML file (or it is incorrect) at agent 13
        # reps element not found in XML file (or it is incorrect) at agent 13
        # Position of agent 13 is too close to some obstacle
        # """
        for cnt, (pos, goal) in enumerate(zip(start_positions, goals)):
            agent = ET.Element("agent")
            # print(cnt)
            # id="1" size="0.3" start.xr="130" start.yr="176" goal.xr="116" goal.yr="243"
            agent.set('id', f'{cnt}')
            agent.set('size', f'{0.01}')

            agent.set('start.xr', f'{pos[0]}')
            agent.set('start.yr', f'{pos[1]}')
            agent.set('goal.xr', f'{goal[0]}')
            agent.set('goal.yr', f'{goal[1]}')

            agents.append(agent)

        tree.write(self.output_xml_file)
        self.valid = True

    def get_goals(self):
        pass

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        agentdict = {} 

        points_lists = []
        goal_lists = []
        start_lists = []

        for agent in root.findall('./log/agent'):
            agent_id = agent.get('id')
            pathfound = agent.find("path").get('pathfound')
            steps = int(agent.find("path").get('steps'))
            
            agentdict[agent_id] = {'pathfound':pathfound, 'steps':steps}
            init_agent = root.find(f'./agents/agent[@id="{agent_id}"]')
            agentdict[agent_id]['start'] = [float(init_agent.get('start.xr')),float(init_agent.get('start.yr'))]
            agentdict[agent_id]['goal'] = [float(init_agent.get('goal.xr')), float(init_agent.get('goal.yr'))]
            agentdict[agent_id]['stepx'] = []
            agentdict[agent_id]['stepy'] = []
            last_x = agentdict[agent_id]['start'][0]
            last_y = agentdict[agent_id]['start'][1]

            max_iter = 1000
            for i in range(max_iter):
                tmp = agent.find("path").find(f'./step[@number="{i}"]')
                if tmp is not None:
                    last_x = float(tmp.get('xr'))
                    last_y = float(tmp.get('yr'))

                agentdict[agent_id]['stepx'].append(last_x)
                agentdict[agent_id]['stepy'].append(last_y)
                
            points_list = list(zip(agentdict[agent_id]['stepx'], agentdict[agent_id]['stepy']))
            points_lists.append(points_list)
            goal_lists.append(agentdict[agent_id]['goal'])
            start_lists.append(agentdict[agent_id]['start'])

        return start_lists, goal_lists, points_lists

    def get_planing(self, start_positions, goals):
        self.set_agents(start_positions, goals)
        # xmlp = "/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/orca_algo/task_examples_demo/custom_road.xml"
        # result = bind.demo(xmlp, self.num_agent)
        result = bind.demo(self.output_xml_file, self.num_agent)

        nexts = []
        for k, v in result.items():
            nextxr = np.array(v.xr)
            nextyr = np.array(v.yr)
            nextr = np.stack([nextxr, nextyr], axis=1)
            nexts.append(nextr)
        assert len(nexts) > 0
        nexts = np.stack(nexts, axis=1)

        self.next_positions = list(nexts)

        return nexts

    def has_next(self):
        if len(self.next_positions) > 0:
            return True
        else:
            return False
        
    def get_next(self):
        if not self.has_next():
            return None
        # print(len(self.next_positions))
        return self.next_positions.pop(0)

    @property
    def length(self):
        return len(self.next_positions)

    def coord_orca_to_md(self, pos, mask_size=256):
        posx = pos[0] - 50
        posy = mask_size - pos[1] - 50

        return (posx, posy)

    def coord_md_to_orca(self, pos, mask_size=255):
        posx =  pos[0] + 50
        posy = mask_size - pos[1] - 50
        return (posx, posy)

    def _random_starts_and_goals(self, map_mask, num):
        def filter_func(x):
            # positions = [(0, 0)]
            # positions = [(0, 0), (2, 2), (-2, -2), (-2, 2), (2, -2)]
            positions = [(0, 0), (3, 3), (-3, -3), (-3, 3), (3, -3)]
            for pos in positions:
                if map_mask[x[1] + pos[0], x[0] + pos[1]] != 255:
                    return False
            return True

        starts = (np.random.rand(100, 100) * 255).astype(np.int32)
        starts = list(filter(filter_func, starts))[:num]
        starts = [(x[0], 255 - x[1]) for x in starts]

        goals = (np.random.rand(100, 100) * 255).astype(np.int32)
        goals = list(filter(filter_func, goals))[:num]
        goals = [(x[0], 255 - x[1]) for x in goals]
  
        len1 = min(len(starts), len(goals))
        len1 = min(len1, num)

        return starts[:len1], goals[:len1]
    
    def random_starts_and_goals(self, map_mask, num):
        left_num = num
        starts, goals = [], []

        while left_num > 0:
            start, goal = self._random_starts_and_goals(map_mask, left_num)
            left_num -= len(start)
            # print(left_num)
            starts += start
            goals += goal
    
        return starts, goals

# # # planning = OrcaPlanning("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/orca_algo/task_examples_demo/custom_road2_template.xml")
# # # planning.get_planing([[154, 176], [130,176], [140,176]], [[96,138],[116,243], [116,243]])

# # # # <agent id="0" size="0.01" start.xr="110" start.yr="216" goal.xr="119.5" goal.yr="250.5"/>
# planning = OrcaPlanning("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/orca_algo/task_examples_demo/custom_road_template.xml")
# # # planning.get_planing([[110, 216]], [[119.5,250.5]])
# # # for i in range(10):
# # #     a = planning.get_next()
# # #     print(a)
# # pos = planning.coord_md_to_orca([100, 120])
# # pos2 = planning.coord_orca_to_md(pos)

# # print(pos)
# # print(pos2)
# import cv2
# mask = cv2.imread("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/map_mask1.png")[..., 0]
# starts, goals = planning.random_starts_and_goals(mask, 100)
# print(len(starts))