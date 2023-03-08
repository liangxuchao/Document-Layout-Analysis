import pyodbc 

from PIL import Image
import shutil
import random
import shutil

import json as JS
from unittest import skip
import os 
fullannotaPath = r'E:\fyp\publaynet\train.json'
with open(fullannotaPath, "r") as json_file:
    data = JS.load(json_file)

## for publaynet

def generateNewAnnotation(annotaionsStartid,startImg,endImg,images,annotations,outputPath):
    annotaionsStartidnew = annotaionsStartid
    print(annotaionsStartidnew)
    batchdata = {"images":[],"annotations":[],"categories":[]}
    for i in range(startImg,endImg):
        batchdata["images"].append(images[i])
        for x in range(annotaionsStartidnew,len(annotations)):
            if annotations[x]['image_id'] == images[i]['id']:
                batchdata["annotations"].append(annotations[x])
            else: 
                annotaionsStartidnew = x
                break 
        # print(i)   
    batchdata["categories"] = [{"supercategory": "", "id": 1, "name": "text"}, 
 {"supercategory": "", "id": 2, "name": "title"}, 
 {"supercategory": "", "id": 3, "name": "list"}, 
 {"supercategory": "", "id": 4, "name": "table"}, 
 {"supercategory": "", "id": 5, "name": "figure"},
 {"supercategory": "", "id": 6, "name": "fomula"},
 {"supercategory": "", "id": 7, "name": "document"}]
    train_json_string = JS.dumps(batchdata)
    with open(outputPath, 'w') as outfile:
        outfile.write(train_json_string)
    return annotaionsStartidnew

images = tuple(data['images'])
annotations = tuple(data['annotations'])

# annotaionsStartid1 = generateNewAnnotation(0,0,60000,images,annotations,r'E:\fyp\publaynet\train_60000_1.json')
# annotaionsStartid2 = generateNewAnnotation(annotaionsStartid1,60000,120000,images,annotations,r'E:\fyp\publaynet\train_60000_2.json')
# annotaionsStartid3 = generateNewAnnotation(annotaionsStartid2,120000,180000,images,annotations,r'E:\fyp\publaynet\train_60000_3.json')
# annotaionsStartid4 = generateNewAnnotation(annotaionsStartid3,180000,240000,images,annotations,r'E:\fyp\publaynet\train_60000_4.json')
# annotaionsStartid5 = generateNewAnnotation(annotaionsStartid4,240000,300000,images,annotations,r'E:\fyp\publaynet\train_60000_5.json')
# annotaionsStartid6 = generateNewAnnotation(annotaionsStartid5,300000,330000,images,annotations,r'E:\fyp\publaynet\val_60000_1.json')

annotaionsStartid1 = generateNewAnnotation(0,0,5000,images,annotations,r'E:\fyp\publaynet\train_5000_1.json')
annotaionsStartid2 = generateNewAnnotation(annotaionsStartid1,5000,10000,images,annotations,r'E:\fyp\publaynet\train_5000_2.json')
annotaionsStartid3 = generateNewAnnotation(annotaionsStartid2,10000,15000,images,annotations,r'E:\fyp\publaynet\train_5000_3.json')
annotaionsStartid4 = generateNewAnnotation(annotaionsStartid3,15000,20000,images,annotations,r'E:\fyp\publaynet\train_5000_4.json')
annotaionsStartid5 = generateNewAnnotation(annotaionsStartid4,20000,25000,images,annotations,r'E:\fyp\publaynet\train_5000_5.json')
annotaionsStartid6 = generateNewAnnotation(annotaionsStartid5,25000,30000,images,annotations,r'E:\fyp\publaynet\train_5000_6.json')
annotaionsStartid7 = generateNewAnnotation(annotaionsStartid6,30000,35000,images,annotations,r'E:\fyp\publaynet\train_5000_7.json')


annotaionsStartid8 = generateNewAnnotation(annotaionsStartid7,35000,40000,images,annotations,r'E:\fyp\publaynet\train_5000_8.json')
annotaionsStartid9 = generateNewAnnotation(annotaionsStartid8,40000,45000,images,annotations,r'E:\fyp\publaynet\train_5000_9.json')
annotaionsStartid10 = generateNewAnnotation(annotaionsStartid9,45000,50000,images,annotations,r'E:\fyp\publaynet\train_5000_10.json')
annotaionsStartid11 = generateNewAnnotation(annotaionsStartid10,50000,55000,images,annotations,r'E:\fyp\publaynet\train_5000_11.json')


annotaionsStartid12 = generateNewAnnotation(annotaionsStartid11,55000,75000,images,annotations,r'E:\fyp\publaynet\train_20000_12.json')

annotaionsStartid13 = generateNewAnnotation(annotaionsStartid12,75000,78000,images,annotations,r'E:\fyp\publaynet\val_3000_1.json')
