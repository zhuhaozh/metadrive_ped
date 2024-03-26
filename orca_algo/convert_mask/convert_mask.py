from PIL import Image

def png_to_2d_list(fname):
    img = Image.open(fname)
    img = img.convert('L')
    binary_array = img.point(lambda x:0 if x<128 else 1, mode='1')
    binary_list = []
    for y in range(binary_array.size[1]):
        row = []
        for x in range(binary_array.size[0]):
            row.append(1 - binary_array.getpixel((x,y))) ## revert 0 & 1
        binary_list.append(row)
    return binary_list

fname = "/home/PJLAB/fujianglin/Desktop/Arlene/ORCA-algorithm/map_mask2.png"
mylist = png_to_2d_list(fname)


# print(mylist)
# print(len(mylist))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# print(mylist)
ax.grid(color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.3)
img = ax.imshow(mylist, cmap='binary',  extent=[0,len(mylist[0]),0,len(mylist)], origin='upper') #extent=[0,w,0,h],
plt.show()


import xml.etree.ElementTree as ET
from xml.dom import minidom
def prettify(elem):
    rough_str = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_str)
    return reparsed.toprettyxml(indent="   ", encoding="utf-8")

def write_to_xml(grid, width, height, cellsize, filename):
    root = ET.Element('map')

    width_elem = ET.SubElement(root, "width")
    width_elem.text = str(width)

    height_elem = ET.SubElement(root, "height")
    height_elem.text = str(height)

    cellsize_elem = ET.SubElement(root, "cellsize")
    cellsize_elem.text = str(cellsize)

    grid_elem = ET.SubElement(root, "grid")
    for row in grid: 
        row_elem = ET.SubElement(grid_elem,"row")
        row_elem.text = " ".join(str(cell) for cell in row)

    tree = ET.ElementTree(root)
    with open(filename, "wb") as f:
        f.write(prettify(root))
        # tree.write(f, encoding="utf-8", xml_declaration=True, pretty_print=True)

# write_to_xml(mylist, len(mylist[0]), len(mylist), 1, "task_examples_demo/custom_road2.xml")