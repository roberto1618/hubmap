import cv2
import numpy as np
import json
from itertools import chain
import os

def mask_generator(img_polygons):
    mask = np.zeros((512, 512))
    coordinates_join = []
    for p in img_polygons:
        coordinates = p['coordinates'][0]
        coordinates_join.append(coordinates)
    coordinates_vessel = list(chain.from_iterable(coordinates_join))
    for coord in coordinates_vessel:
        mask[coord[1], coord[0]] = 1
    
    mask = mask.astype(np.uint8)# * 255
        
    return mask

folder = './data/labels'
if not os.path.exists(folder):
    os.mkdir(folder)

with open('./data/polygons.jsonl', 'r') as f:
    polygons = [json.loads(line) for line in f]

for i,p in enumerate(polygons):
    img_name = p['id']
    polygons_blood_vessel = [d for d in p['annotations'] if d['type'] == 'blood_vessel']
    mask = mask_generator(polygons_blood_vessel)
    cv2.imwrite(f'./data/labels/{img_name}.tif', mask)