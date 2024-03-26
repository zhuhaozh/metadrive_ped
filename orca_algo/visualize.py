import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET 
import matplotlib.patches as pathces
import numpy as np
import matplotlib.colors as mcolors
base_colors = [(0.7,0.2,0.3),(0.1,0.5, 0.8),(0.9,0.6,0.2),
               (0.3,.8,.1),(.5,.4,.7),(.8,.3,.5),
               (.2,.7,.9),(.6,.9,.4),(.4,.1,.6),
               (.9,.2,.7)]
colors1 = []
colors2 = []
colors3 = []
colors4 = []
for color in base_colors:
    cmap=plt.get_cmap('coolwarm')
    colors1.append(cmap(1/4)[:3])
    colors2.append(cmap(2/4)[:3])
    colors3.append(cmap(3/4)[:3])
    colors4.append(cmap(4/4)[:3])
    # for i in range(4):
    #     colors.append(cmap(i/4)[:3])
colors = base_colors+ colors1+colors2+colors3+colors4   

# color_num=50
# #### color method1
# # import random
# # # colors = ['r','b','g','orange','purple','cyan','magenta','yellow','pink','brown']
# # colors = [mcolors.to_hex(color) for color in mcolors.TABLEAU_COLORS.values()]
# # while len(colors) < color_num:
# #     colors +=  [mcolors.to_hex(color) for color in mcolors.TABLEAU_COLORS.values()]
# # random.shuffle(colors)
# # print(colors)
# colors = ['#e377c2', '#2ca02c', '#17becf', '#2ca02c', '#9467bd', '#ff7f0e', '#2ca02c', '#8c564b', '#ff7f0e', '#1f77b4', '#7f7f7f', '#1f77b4', '#ff7f0e', '#bcbd22', '#ff7f0e', '#17becf', '#bcbd22', '#8c564b', '#7f7f7f', '#ff7f0e', '#e377c2', '#bcbd22', '#9467bd', '#8c564b', '#1f77b4', '#d62728', '#17becf', '#2ca02c', '#7f7f7f', '#e377c2', '#9467bd', '#d62728', '#17becf', '#17becf', '#9467bd', '#1f77b4', '#e377c2', '#2ca02c', '#8c564b', '#7f7f7f', '#bcbd22', '#7f7f7f', '#bcbd22', '#d62728', '#e377c2', '#d62728', '#9467bd', '#1f77b4', '#d62728', '#8c564b']


def parse_xml(root):
    w = int(root.find('./map/width').text)
    h = int(root.find('./map/height').text)
    obs = []

    for obs_ele in root.findall('./obstacles/obstacle'):
        # print('..',obs_ele)
        vertices = []
        for vertex_ele in obs_ele.findall('./vertex'):
            # x = int(vertex_ele.find('xr').text)
            # y = int(vertex_ele.find('yr').text)
            xr = float(vertex_ele.get('xr'))
            yr = float(vertex_ele.get('yr'))
            vertices.append((xr,yr))
        obs.append(vertices)

    grid = []
    for row in root.find('./map/grid').findall('row'):
        row_value = list(map(int, row.text.split()))
        grid.append(row_value)
    import numpy as np
    grid = np.flipud(grid)
    return w, h, obs, grid


# xml_file = '/home/PJLAB/fujianglin/Desktop/Arlene/ORCA-algorithm/task_examples/empty_task_10_log.xml'
# xml_file = '/home/PJLAB/fujianglin/Desktop/Arlene/ORCA-algorithm/task_examples/0_task_10_log.xml'
# xml_file = '/home/PJLAB/fujianglin/Desktop/Arlene/ORCA-algorithm/task_examples/1_task_fjl_3_log.xml'
# for k in [0,10,20,30,40]:
for k in [2]:
    # fname = f'1_task_fjl{k}_5'
    # fname= f'0_task_{k}-{k+9}_10' #'1_task_fjl1_5' #
    # fname = f'0_task_ecbs_{k}'
    fname =f"custom_road2_{k}"
    print(f'processing {fname}')
    xml_file = "/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/orca_algo/task_examples_demo/custom_road_1_log.xml"
    # xml_file = f'/home/PJLAB/fujianglin/Desktop/Arlene/ORCA-algorithm/task_examples_demo/{fname}_log.xml'
    output_name = f'../video_result/{fname}_newcolor.mp4'
    visualize = True #False #True

    tree = ET.parse(xml_file)
    root = tree.getroot()
    w, h, obs, grid = parse_xml(root)

    ## get agents ids
    agentdict = {} 
            # {id: 
            #   {
            #       start=[start.xr, start.yr], 
            #       goal = [goal.xr, goal.yr],
            #       pathfound = T/F,
            #       steps = #,
            #       step_x = [],
            #       step_y = [],
            #   }
            # }

    points_lists = []
    fixed_goals = []
    start_lists = []

    for agent in root.findall('./log/agent'):
        agent_id = agent.get('id')
        # for child in agent:
            # print(f'--{agent_id},{child.tag}, {child.text}')
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

        max_iter = 3000 #steps+1 if steps < 1000 else 1000+1 #1000 #
        for i in range(max_iter):
            
            tmp = agent.find("path").find(f'./step[@number="{i}"]')
            if tmp is not None:
                last_x = float(tmp.get('xr'))
                last_y = float(tmp.get('yr'))
            # else:
            #     pass #pad with previous value
            # print(agent_id,i,tmp)
            agentdict[agent_id]['stepx'].append(last_x)
            agentdict[agent_id]['stepy'].append(last_y)
            
        points_list = list(zip(agentdict[agent_id]['stepx'], agentdict[agent_id]['stepy']))
        points_lists.append(points_list)
        fixed_goals.append(agentdict[agent_id]['goal'])
        start_lists.append(agentdict[agent_id]['start'])

    fig, ax = plt.subplots()

    ax.grid(color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.3)
    img = ax.imshow(grid, cmap='binary', extent=[0,w,0,h], origin='lower')
    # plt.show()
    # assert False
    # plt.imshow(grid, cmap='binary', interpolation='nearest', origin='lower')
    # plt.cm.get_cmap('gray').set_bad(color='gray'


    import matplotlib.animation as animation
    ### FOR SINGLE AGENT
    # tmp_id='5'
    # points_list = list(zip(agentdict[tmp_id]['stepx'], agentdict[tmp_id]['stepy']))
    # def init():
    #     global fixed_goal
    #     fixed_goal = ax.scatter(agentdict[tmp_id]['goal'][0],agentdict[tmp_id]['goal'][1], c='red', marker='*')
    #     return [img, fixed_goal]
    # def update(frame):
    #     current = points_list[frame]
    #     x, y = current
    #     scatter = ax.scatter(x,y, c='red', marker='o')
    #     return [img, scatter, fixed_goal]
    # ani = animation.FuncAnimation(fig, update, frames=len(agentdict[tmp_id]['stepx']), init_func=init, blit=True)
    # plt.show()

    ### FOR MULTIPLE AGENT
    fixed_colors = [colors[i] for i in range(len(fixed_goals))]
    fixed_goal = ax.scatter([p[0] for p in fixed_goals], [p[1] for p in fixed_goals], marker='x', color=fixed_colors)
    fixed_start = ax.scatter([p[0] for p in start_lists], [p[1] for p in start_lists], marker='o', color=fixed_colors)
    scatters = [ax.scatter([],[], label=f'Agent {i+1}', color=colors[i]) for i in range(len(points_lists))]

    def update(frame):
        for i, points_list in enumerate(points_lists):
            if frame < len(points_list):
                x, y = points_list[frame]
                scatters[i].set_offsets((x,y))
            else:
                scatters[i].set_offsets([],[])
        return scatters
    ani = animation.FuncAnimation(fig, update, frames=len(max(points_lists,key=len)), blit=True, interval=10)
    # ani.save('../video_resulttest.gif', writer='pillow')
    if visualize:
        plt.show()
    else:
        ani.save(output_name, writer='ffmpeg')
