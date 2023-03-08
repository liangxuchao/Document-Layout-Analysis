import json as JS
from unittest import skip
import os 

#fullannotaPath =  r"E:\fyp\warpdoc\withcoco\val_wrap.json"
fullannotaPath =  r"E:\fyp\doclaynet\COCO\train_doclaynet.json"


with open(fullannotaPath, "r") as json_file:
    data = JS.load(json_file)

count = 0

# wrap doc
# for anno in data['annotations']:
#     count = count + 1
#     print(count)
#     if anno['category_id'] == 3:
#         anno['category_id'] = 4
#     elif anno['category_id'] == 4:
#         anno['category_id'] = 3
#     elif anno['category_id'] == 7:
#         anno['category_id'] = 6 
#     elif anno['category_id'] == 9:
#         anno['category_id'] = 7


# doclaynet doc

def annotation_document(img):
    idstart = 10000000
    obj = {
      "id": idstart + img['id'],
      "image_id": img['id'],
      "category_id": 12,
      "bbox": [
        0,
        0,
        img['width'],
        img['height']
      ],
      "segmentation": [
        [
          0,
          0,
          0,
          img['height'],
          img['width'],
          img['height'],
          img['width'],
          0
        ]
      ],
      "area": img['width'] * img['height'],
      "iscrowd": 0,
      "precedence": 0
    }
    return obj


def changeAnnotationID(annotaionsStartid,startImg,endImg,images,annotations,outputPath):
    annotaionsStartidnew = annotaionsStartid
    print(annotaionsStartidnew)
    batchdata = {"images":[],"annotations":[],"categories":[]}
    for i in range(startImg,endImg):
        imageadd = False
        for x in range(annotaionsStartidnew,len(annotations)):
            
            if annotations[x]['image_id'] == images[i]['id']:
            #     if annotations[x]['category_id'] in [3,7,9,10,11,8]:
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
            batchdata["annotations"].append(annotation_document(images[i]))
            batchdata["images"].append(images[i])
        # print(i)   
#     batchdata["categories"] = [{"supercategory": "", "id": 1, "name": "text"}, 
#  {"supercategory": "", "id": 2, "name": "title"}, 
#  {"supercategory": "", "id": 3, "name": "list"}, 
#  {"supercategory": "", "id": 4, "name": "table"}, 
#  {"supercategory": "", "id": 5, "name": "figure"},
#  {"supercategory": "", "id": 6, "name": "fomula"},
#  {"supercategory": "", "id": 7, "name": "document"}]
    batchdata["categories"] = [{
      "supercategory": "Caption",
      "id": 1,
      "name": "Caption"
    },
    {
      "supercategory": "Footnote",
      "id": 2,
      "name": "Footnote"
    },
    {
      "supercategory": "Formula",
      "id": 3,
      "name": "Formula"
    },
    {
      "supercategory": "List-item",
      "id": 4,
      "name": "List-item"
    },
    {
      "supercategory": "Page-footer",
      "id": 5,
      "name": "Page-footer"
    },
    {
      "supercategory": "Page-header",
      "id": 6,
      "name": "Page-header"
    },
    {
      "supercategory": "Picture",
      "id": 7,
      "name": "Picture"
    },
    {
      "supercategory": "Section-header",
      "id": 8,
      "name": "Section-header"
    },
    {
      "supercategory": "Table",
      "id": 9,
      "name": "Table"
    },
    {
      "supercategory": "Text",
      "id": 10,
      "name": "Text"
    },
    {
      "supercategory": "Title",
      "id": 11,
      "name": "Title"
    },{"supercategory": "", "id": 12, "name": "document"}]
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

annotaionsStartid1 = changeAnnotationID(0,0,69375,images,annotations,r'E:\fyp\doclaynet\COCO\train_69375_1.json')

# annotaionsStartid1 = changeAnnotationID(0,0,6300,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_1.json')
# annotaionsStartid2 = changeAnnotationID(annotaionsStartid1,6300,12600,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_2.json')
# annotaionsStartid3 = changeAnnotationID(annotaionsStartid2,12600,18900,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_3.json')
# annotaionsStartid4 = changeAnnotationID(annotaionsStartid3,18900,25200,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_4.json')
# annotaionsStartid5 = changeAnnotationID(annotaionsStartid4,25200,31500,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_5.json')
# annotaionsStartid6 = changeAnnotationID(annotaionsStartid5,31500,37800,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_6.json')
# annotaionsStartid7 = changeAnnotationID(annotaionsStartid6,37800,44100,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_7.json')
# annotaionsStartid8 = changeAnnotationID(annotaionsStartid7,44100,50400,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_8.json')
# annotaionsStartid9 = changeAnnotationID(annotaionsStartid8,50400,56700,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_9.json')
# annotaionsStartid10 = changeAnnotationID(annotaionsStartid9,56700,63000,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_10.json')
# annotaionsStartid11 = changeAnnotationID(annotaionsStartid10,63000,69300,images,annotations,r'E:\fyp\doclaynet\COCO\train_6300_11.json')

# annotaionsStartid1 = changeAnnotationID(0,0,500,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_1.json')
# annotaionsStartid2 = changeAnnotationID(annotaionsStartid1,500,1000,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_2.json')
# annotaionsStartid3 = changeAnnotationID(annotaionsStartid2,1000,1500,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_3.json')
# annotaionsStartid4 = changeAnnotationID(annotaionsStartid3,1500,2000,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_4.json')
# annotaionsStartid5 = changeAnnotationID(annotaionsStartid4,2000,2500,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_5.json')
# annotaionsStartid6 = changeAnnotationID(annotaionsStartid5,2500,3000,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_6.json')
# annotaionsStartid7 = changeAnnotationID(annotaionsStartid6,3000,3500,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_7.json')
# annotaionsStartid8 = changeAnnotationID(annotaionsStartid7,3500,4000,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_8.json')
# annotaionsStartid9 = changeAnnotationID(annotaionsStartid8,4000,4500,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_9.json')
# annotaionsStartid10 = changeAnnotationID(annotaionsStartid9,4500,5000,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_10.json')
# annotaionsStartid11 = changeAnnotationID(annotaionsStartid10,5000,5500,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_11.json')


# annotaionsStartid12 = changeAnnotationID(annotaionsStartid11,55000,60000,images,annotations,r'E:\fyp\doclaynet\COCO\train_5000_12.json')
# annotaionsStartid13 = changeAnnotationID(annotaionsStartid12,60000,60500,images,annotations,r'E:\fyp\doclaynet\COCO\val_500_1.json')

# print("remove start")
# for item in data['annotations'].copy():
#     count = count + 1
#     print(count)
#     if item['category_id'] not in [3,4,7,9,10,11,8]:
#         data['annotations'].remove(item)
# print("remove end")
# for index,item in enumerate(data['annotations']):
#     if item['category_id'] == 3:
#         data['annotations'][index]['category_id'] = 6
#     elif item['category_id'] == 4:
#         data['annotations'][index]['category_id'] = 3
#     elif item['category_id'] == 7:
#         data['annotations'][index]['category_id'] = 5 
#     elif item['category_id'] == 9:
#         data['annotations'][index]['category_id'] = 4
#     elif item['category_id'] ==10:
#         data['annotations'][index]['category_id'] = 1
#     elif item['category_id'] ==11 or item['category_id'] ==8:
#         data['annotations'][index]['category_id'] = 2


# data['categories'] = [{"supercategory": "", "id": 1, "name": "text"}, 
#  {"supercategory": "", "id": 2, "name": "title"}, 
#  {"supercategory": "", "id": 3, "name": "list"}, 
#  {"supercategory": "", "id": 4, "name": "table"}, 
#  {"supercategory": "", "id": 5, "name": "figure"},
#  {"supercategory": "", "id": 6, "name": "fomula"},
#  {"supercategory": "", "id": 7, "name": "document"}]

# outputpath =  r"E:\fyp\doclaynet\COCO\val_60_1.json"
# with open(outputpath, "w") as jsonFile:
#     JS.dump(data, jsonFile)

