import xml.etree.ElementTree as ET
import sys
from metadrive.engine.logger import get_logger

logger = get_logger()

sys.path.insert(0, './orca_algo/build')
import bind
import numpy as np
import math



from PIL import Image, ImageOps
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import math

def slope(p1, p2):
    if (p2[0]- p1[0]) == 0: return np.inf
    elif (p2[1]- p1[1]) == 0: return 0
    else: return (p2[1]- p1[1]) / p2[0]- p1[0]

def is_collinear(p1, p2, p3):
    return np.abs(slope(p1,p2) - slope(p2,p3)) < 0.1

def remove_middle_points(pts):
    if len(pts)<3: return pts
    filtered = [pts[0]]
    prev_pt = pts[0]

    for i in range(1, len(pts)-1):
        next_pt = pts[i+1]
        if is_collinear(prev_pt, pts[i], next_pt): continue
        filtered.append(pts[i])

    filtered.append(pts[-1]) # add last point
    return filtered

def find_tuning_point(contour, h):
    unique_pt = []
    filtered_contour = []
    ppp = len(contour)
    for i, (y,x) in enumerate(contour):
        if len(unique_pt) == 0 or (x != unique_pt[-1][0] and (h - 1 - y) != unique_pt[-1][1]):
            y = h - 1 - y
            unique_pt.append([x,y])
    prev_len = len(unique_pt)
    unique_pt = remove_middle_points(unique_pt)
    # print(ppp, '   . ', prev_len, ' . ', len(unique_pt))
    return np.array(unique_pt)

def mask_to_2d_list(mask, extend=False, upsample=1):
    img = Image.fromarray(mask)
    if upsample>1:
        neww = img.width * upsample
        newh = img.height * upsample
        img = img.resize((neww, newh), Image.BICUBIC)
    if extend:
        img = ImageOps.expand(img, border=2, fill='black')

    img = img.convert('L')
    binary_array = img.point(lambda x:0 if x<128 else 1, mode='1')
    h = binary_array.size[1]
    w = binary_array.size[0]
    binary_list = []
    for y in range(h):
        row = []
        for x in range(w):
            row.append(1 - binary_array.getpixel((x,y))) ## revert 0 & 1
        binary_list.append(np.array(row))
    binary_list = np.array(binary_list)
    return binary_list, h, w

def prettify(elem):
    rough_str = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_str)
    return reparsed.toprettyxml(indent="   ", encoding="utf-8")

def write_to_xml(grid, width, height, cellsize, flipped_contours, agentdict, filename):
    root = ET.Element('root')


    agents = ET.SubElement(root,'agents')
    agents.set('number', str(len(agentdict['agent'])))
    agents.set('type', agentdict['type'])
    default_parameters = ET.SubElement(agents,'agents',
                                       size="0.3", 
                                       movespeed="1", 
                                       agentsmaxnum="10", 
                                       timeboundary="5.4", 
                                       sightradius="3.0", 
                                       timeboundaryobst="33")

    for agent in agentdict['agent']:
        tmpagent = ET.SubElement(agents,"agent",
                            id=agent["id"],
                            size=agent["size"],
                            **{"start.xr":str(agent["start.xr"]),
                            "start.yr":str(agent["start.yr"]),
                            "goal.xr":str(agent["goal.xr"]), 
                            "goal.yr":str(agent["goal.yr"])})
    # agent2 = ET.SubElement(agents,'agent',
    #                        id="1",
    #                        size="0.1",
    #                        **{"start.xr":"323.5",
    #                        "start.yr":"123.5",
    #                        "goal.xr":"239.5", 
    #                        "goal.yr":"160.5"})
    obstacles = ET.SubElement(root,'obstacles')
    obstacles.set('number', str(len(flipped_contours)))
    for k, contour in enumerate(flipped_contours):

        obstacle = ET.SubElement(obstacles, 'obstacle')
        for pt in contour:
            xr, yr = pt
            vertex = ET.SubElement(obstacle,'vertex')
            vertex.set('xr', str(int(xr)))
            vertex.set('yr', str(int(yr)))


    map = ET.SubElement(root, "map")
    width_elem = ET.SubElement(map, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(map, "height")
    height_elem.text = str(height)
    cellsize_elem = ET.SubElement(map, "cellsize")
    cellsize_elem.text = str(cellsize)
    grid_elem = ET.SubElement(map, "grid")
    for row in grid: 
        row_elem = ET.SubElement(grid_elem,"row")
        row_elem.text = " ".join(str(cell) for cell in row)     


    algo = ET.SubElement(root, 'algorithm')
    searchtype = ET.SubElement(algo, 'searchtype')
    searchtype.text = 'thetastar'
    breakingties = ET.SubElement(algo, 'breakingties')
    breakingties.text = '0'
    allowsqueeze = ET.SubElement(algo, 'allowsqueeze')
    allowsqueeze.text = 'false'
    cutcorners = ET.SubElement(algo, 'cutcorners')
    cutcorners.text = 'false'
    hweight = ET.SubElement(algo, 'hweight')
    hweight.text = '1'
    timestep = ET.SubElement(algo, 'timestep')
    timestep.text = '0.1'
    delta = ET.SubElement(algo, 'delta')
    delta.text = '0.1'
    trigger = ET.SubElement(algo, 'trigger')
    trigger.text = 'speed-buffer'
    mapfnum = ET.SubElement(algo, 'mapfnum')
    mapfnum.text = '3'

    tree = ET.ElementTree(root)
    with open(filename, "wb") as f:
        f.write(prettify(root))
        # tree.write(f, encoding="utf-8", xml_declaration=True, pretty_print=True)

    
class OrcaPlanning:
    def __init__(self, template_xml_file=None, map_mask=None):
        # self.template_xml_file = template_xml_file
        if template_xml_file is None:
            self.template_xml_file = "output/output_xml/output_template.xml"
        else:
            self.template_xml_file = template_xml_file

        self.output_xml_file = 'output/output_xml/output_template_agent.xml'
        self.valid = False
        self.num_agent = -1
        self.next_positions = []
        self.prev_start_positions, self.prev_goals = None, None

    def generate_template_xml(self, mask):
        upsample = 1  # 1, or other positive integer
        extend = False # extend image's four side by 2 pixels or not
        cellsize = 1

        # img_name = f"test_walkable_mask{ll}" #'map_mask'
        # fname = os.path.join(folder,f'{img_name}.png') 

        agentdict = {"type":"orca-par-ecbs", 
                    "agent": []}

        mylist, h, w = mask_to_2d_list(mask, extend=extend, upsample=upsample)
        contours = measure.find_contours(mylist, 0.5, positive_orientation='high') #extend_mylist

        flipped_contours = []
        for contour in contours:
            contour = find_tuning_point(contour, h) #h+4
            flipped_contours.append(contour)

        write_to_xml(mylist, w, h, cellsize, flipped_contours, agentdict, self.template_xml_file)

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
            agent.set('size', f'{0.3}')

            agent.set('start.xr', f'{pos[0]}')
            agent.set('start.yr', f'{pos[1]}')
            agent.set('goal.xr', f'{goal[0]+0.5}') # magic number 
            agent.set('goal.yr', f'{goal[1]+0.5}')

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

    def get_planning(self, start_positions, goals):

        def get_speed(positions):
            pos1 = positions[:-1]
            pos2 = positions[1:]

            pos_delta = pos2 - pos1
            speed = np.linalg.norm(pos_delta, axis=2)
            speed = np.concatenate([np.zeros((1, len(start_positions))), speed], axis=0)
            return list(speed)


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

        if len(nexts) == 0:
            logger.warning("ORCA planning error, return None")
            return None

        nexts = np.stack(nexts, axis=1)
        self.next_positions = list(nexts)
        self.speed = get_speed(nexts)
        return nexts

    def has_next(self):
        if len(self.next_positions) > 0:
            return True
        else:
            return False
        
    def get_next(self, return_speed=False):
        if not self.has_next():
            return None

        if not return_speed:
            return self.next_positions.pop(0)
        else:
            return self.next_positions.pop(0), self.speed.pop(0)

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
  
    def _random_points(self, map_mask, num):
        def in_walkable_area(x):
            # positions = [(0, 0)]
            # positions = [(0, 0), (2, 2), (-2, -2), (-2, 2), (2, -2)]
            positions = [(0, 0), (3, 3), (-3, -3), (-3, 3), (3, -3)]
            try:
                for pos in positions:
                    if map_mask[x[1] + pos[0], x[0] + pos[1]] != 255:
                        return False
                return True
            except Exception:
                return False
                    
        def is_close_to_points(x, pts, filter_rad=5):
            min_dist = 9999
            for pt in pts:
                min_dist = min(min_dist, math.dist(x, pt))
            
            if min_dist < filter_rad:
                return True
            else:
                return False

        pts = []
        h, w = map_mask.shape
        while len(pts) < num:
            pt = (np.random.randint(0, w - 1), np.random.randint(0, h - 1))
            if not in_walkable_area(pt):
                continue
            if len(pts) > 0 and is_close_to_points(pt, pts):
                continue
            pts.append(pt)
        # pts = [(x[0], 255 - x[1]) for x in pts]
        pts = [(x[0], h - 1 - x[1]) for x in pts]

        return pts

    def random_starts_and_goals(self, map_mask, num):
        starts = self._random_points(map_mask, num)
        goals = self._random_points(map_mask, num)

        return starts, goals


# planning = OrcaPlanning("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/orca_algo/task_examples_demo/custom_road2_template.xml")

# # # # # <agent id="0" size="0.01" start.xr="110" start.yr="216" goal.xr="119.5" goal.yr="250.5"/>
# import cv2

# planning = OrcaPlanning("tmp/test_template.xml")
# mask = cv2.imread("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/tmp/map_mask1.png")[..., 0]
# planning.generate_template_xml(mask)
# print("finished")
# starts, goals = planning.random_starts_and_goals(mask, 5)

# planning.get_planning(starts, goals)