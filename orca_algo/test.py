import sys
sys.path.insert(0, '../build')
import bind

import xml.etree.ElementTree as ET

if __name__ == "__main__":
    
    xml_path = "/home/PJLAB/fujianglin/Desktop/Arlene/ORCA-algorithm/task_examples_demo/custom_road.xml"
    agentnum = 1

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ### changed agent
    agent_id = 0
    

    ### change specific value in xml
    init_agent = root.find(f'./agents/agent[@id="{agent_id}"]')
    print(float(init_agent.get('start.xr')))
    if init_agent is not None and 'start.xr' in init_agent.attrib:
        init_agent.set('start.xr','110')
    print(float(init_agent.get('start.xr'))) 

    ### overwrite xml with changed value
    tree.write(xml_path) 

    assert False
    result = bind.demo(xml_path, agentnum)
    print(result.keys())
    # print(result.values())
    for k, v in result.items():
        print(k)
        print("xr: ", [f'{num:5f}' for num in v.xr])
        print("yr: ", [round(num,5) for num in v.yr])
        print("nextxr: ", v.nextxr, len(v.nextxr))
        print("nextyr: ", v.nextyr)
        print("foundpath: ", v.foundpath)
        print("total_step: ", v.total_step)