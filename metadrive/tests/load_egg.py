import re
import numpy as np
import panda3d.egg as egg


# https://github.com/OlafHaag/bvh-toolbox/tree/main
# https://github.com/OlafHaag/bvh-toolbox/blob/main/src/bvhtoolbox/convert/bvh2egg.py

# https://docs.panda3d.org/1.10/python/reference/panda3d.egg#module-panda3d.egg



"""
<S$Anim> i { <V> { 1.0 } }
<S$Anim> j { <V> { 1.0 } }
<S$Anim> k { <V> { 1.0 } }
<S$Anim> p { <V> { 2e-06 -1e-06 3e-06 5e-06 1e-06 2e-06 1e-06 3e-06 3e-06 1e-06 2e-06 3e-06 2e-06 -0.0 3e-06 6e-06 2e-06 3e-06 4e-06 4e-06 -0.0 3e-06 4e-06 5e-06 3e-06 6e-06 4e-06 -0.0 3e-06 1e-06 2e-06 2e-06 0.0 1e-06 -0.0 3e-06 2e-06 3e-06 3e-06 4e-06 7e-06 1e-06 4e-06 2e-06 2e-06 3e-06 0.0 0.0 2e-06 -0.0 0.0 4e-06 7e-06 5e-06 0.0 2e-06 2e-06 2e-06 2e-06 2e-06 1e-06 } }
<S$Anim> r { <V> { -1e-06 2e-06 -0.0 0.0 -3e-06 0.0 1e-06 0.0 -0.0 0.0 0.0 -0.0 -1e-06 -0.0 1e-06 2e-06 -0.0 -1e-06 -3e-06 1e-06 2e-06 -1e-06 0.0 -3e-06 -1e-06 -1e-06 -1e-06 -1e-06 0.0 -1e-06 -1e-06 1e-06 0.0 0.0 2e-06 1e-06 -1e-06 -1e-06 -1e-06 1e-06 -2e-06 -4e-06 3e-06 -3e-06 4e-06 1e-06 -1e-06 1e-06 -2e-06 -1e-06 2e-06 2e-06 -3e-06 0.0 0.0 0.0 -1e-06 0.0 1e-06 0.0 0.0 } }
<S$Anim> h { <V> { 1e-06 -2e-06 -1e-06 0.0 -0.0 1e-06 9e-06 -1e-06 -3e-06 2e-06 1e-06 2e-06 -3e-06 0.0 2e-06 7e-06 3e-06 -3e-06 3e-06 4e-06 -4e-06 1e-06 -1e-06 -0.0 3e-06 -4e-06 3e-06 -1e-06 4e-06 1e-06 -5e-06 0.0 1e-06 6e-06 4e-06 2e-06 -1e-06 -3e-06 3e-06 2e-06 1e-06 1e-06 4e-06 -2e-06 3e-06 2e-06 -1e-06 1e-06 5e-06 3e-06 6e-06 -2e-06 1e-06 2e-06 2e-06 1e-06 -1e-06 -1e-06 4e-06 -1e-06 2e-06 } }
<S$Anim> x { <V> { 0.002125 } }
<S$Anim> y { <V> { -0.025338 } }
<S$Anim> z { <V> { -0.096714 } }
"""

import re


def load_egg(egg_file):
    with open(egg_file) as f:
        lines = f.readlines()

    bones_dict = {}
    prev_is_anim = False
    cnt11 = 0
    for cnt in range(len(lines)):
        line = lines[cnt].strip()
        if prev_is_anim:
            bones_dict[bone_name] = bone_dict

        if line.startswith("<Table>"):
            sp = line.split(" ")
            if len(sp) == 3 and sp[1] != "\"<skeleton>\"":
                bone_dict = {"i":[], "j":[], "k":[], 
                             "p":[], "r":[], "h":[], 
                             "x":[], "y":[], "z":[]}
                bone_name = sp[1]
                cnt11 += 1
        
        if line.startswith("<S$Anim>"):
            prev_is_anim = True
            t = re.findall(r"\<S\$Anim\> (.) ", line)[0]
            vals = re.findall(r"\{ <V> \{ .* \} \}", line)[0].replace("{ <V> { ", "").replace(" } }", "").split(" ")

            vals = [float(val) for val in vals]
            bone_dict[t] = vals
        else:
            prev_is_anim = False
    
    # print(cnt11, len(bones_dict))
    return convert_to_list(bones_dict)


def convert_to_list(bones_dict):
    num_frame = 60
    frames = []
    for i in range(num_frame):
        frame_dict = {}
        for bone, vals in bones_dict.items():
            if bone not in frame_dict:
                frame_dict[bone] = {}
            
            # print(len(vals['r']))
            if len(vals['p']) != 0:
                if len(vals['p']) == 1:
                    frame_dict[bone]['rotation'] = (vals['r'][0], vals['p'][0], vals['h'][0])
                else:
                    frame_dict[bone]['rotation'] = (vals['r'][i], vals['p'][i], vals['h'][i])
            else:
                frame_dict[bone]['rotation'] = None

            if len(vals['x']) != 0:
                if len(vals['x']) == 1:
                    frame_dict[bone]['position'] = (vals['x'][0], vals['y'][0], vals['z'][0])
                else:
                    frame_dict[bone]['position'] = (vals['x'][i], vals['y'][i], vals['z'][i])
            else:
                frame_dict[bone]['position'] = None

        frames.append(frame_dict)
    return frames

if __name__ == '__main__':
    bones_dict = load_egg("/home/PJLAB/zhuhao/workspace/MDs/metadrive_ped/tmp/walk60.egg")