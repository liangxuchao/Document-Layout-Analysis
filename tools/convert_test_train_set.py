# Program to read JSON file 
# and generate its XML file
   
# Importing json module and xml
# module provided by python
import json as JS
from unittest import skip
import xml.etree.ElementTree as ET

import os 
#"../datasets/publaynet_full/full.json"
#"../datasets/publaynet_full/train"
# imagePath = "../datasets/publaynet_full/train"
# annotaPath = "../datasets/publaynet_full/full.json"
imagePath = "../datasets/publaynet_full/val"
annotaPath = "../datasets/publaynet_full/full.json"
train_files = os.listdir(imagePath)   
# fullmetajson = open("../datasets/publaynet_full/full.json", "r")

newImage = []
newAnnotation = []
newCategory = []
count = 0
with open(annotaPath, "r") as json_file:
    data = JS.load(json_file)
    for i in data['images']:
        if i["file_name"] in train_files:
            count = count + 1
            print(count)
            newImage.append(i)

            for y in data["annotations"]:
                if y["image_id"] == i["id"]:
                    newAnnotation.append(y)
                    break
        else:
            continue
    
newData = {"images":newImage,"annotations":newAnnotation,"categories":data['categories']}

json_string = JS.dumps(newData)


outputpath = '../datasets/publaynet_full/val.json'
# outputpath = '../datasets/publaynet_full/train.json'
with open(outputpath, 'w') as outfile:
    outfile.write(json_string)
