import pyodbc 

from PIL import Image
import shutil
import random
import shutil

import json as JS
from unittest import skip
import os 
fullannotaPath = r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotateN15.json'
with open(fullannotaPath, "r") as json_file:
    data = JS.load(json_file)


def changeAnnotationID(annotaionsStartid,startImg,endImg,images,annotations,outputPath):
    annotaionsStartidnew = annotaionsStartid
    print(annotaionsStartidnew)
    batchdata = {"images":[],"annotations":[],"categories":[]}
    for i in range(startImg,endImg):
        imageadd = False
        for x in range(annotaionsStartidnew,len(annotations)):
            
            if annotations[x]['image_id'] == images[i]['id']:
                if annotations[x]['category_id'] in [12]:
            #     # if annotations[x]['category_id'] in [3]:
            #         annotationNew = annotations[x]
            #         if annotations[x]['category_id'] == 3:
            #             annotationNew['category_id'] = 6
            #         elif annotations[x]['category_id'] == 4:
            #             annotationNew['category_id'] = 3
            #         elif annotations[x]['category_id'] == 7:
            #             annotationNew['category_id'] = 5 
            #         elif annotations[x]['category_id'] == 9:
            #             annotationNew['category_id'] = 4
            #         elif annotations[x]['category_id'] ==10:
            #             annotationNew['category_id'] = 1
            #         elif annotations[x]['category_id'] ==11 or annotations[x]['category_id'] ==8:
            #             annotationNew['category_id'] = 2
                    imageadd = True
                    batchdata["annotations"].append(annotations[x])
                
            else: 
                annotaionsStartidnew = x
                break 
        
        if imageadd:
            batchdata["images"].append(images[i])
    batchdata["categories"] = [{
        "supercategory": "", "id": 12, "name": "document"}]
    train_json_string = JS.dumps(batchdata)
    with open(outputPath, 'w') as outfile:
        outfile.write(train_json_string)
    print("output")
    print(len(batchdata["images"]))
    print(len(batchdata["annotations"]))
    return annotaionsStartidnew

images = tuple(data['images'])
annotations = tuple(data['annotations'])

print(len(images))

annotaionsStartid1 = changeAnnotationID(0,0,69103,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotateN15_1c.json')
