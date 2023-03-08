import json as JS
from unittest import skip
import os 

#fullannotaPath =  r"E:\fyp\warpdoc\withcoco\val_wrap.json"
fullannotaPath =  r"E:\fyp\warpdoc\withcoco\wrap-rotate_12cat.json"


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


def changeAnnotationID(annotaionsStartid,startImg,endImg,images,annotations,outputPath):
    annotaionsStartidnew = annotaionsStartid
    print(annotaionsStartidnew)
    batchdata = {"images":[],"annotations":[],"categories":[]}
    for i in range(startImg,endImg):
        batchdata["images"].append(images[i])
        for x in range(annotaionsStartidnew,len(annotations)):
            
            if annotations[x]['image_id'] == images[i]['id']:
                # if annotations[x]['category_id'] in [1,2,3,4,5,7,9]:
                # if annotations[x]['category_id'] in [1,2,3,5,7,9]:
                    annotationNew = annotations[x]
                    
                    # if annotations[x]['category_id'] == 1:
                    #     annotationNew['category_id'] = 10
                    # elif annotations[x]['category_id'] == 2:
                    #     annotationNew['category_id'] = 11 
                    # elif annotations[x]['category_id'] == 3:
                    #     annotationNew['category_id'] = 9
                    # elif annotations[x]['category_id'] == 4:
                    #     annotationNew['category_id'] = 3
                    # elif annotations[x]['category_id'] == 5:
                    #     annotationNew['category_id'] = 7 
                    # elif annotations[x]['category_id'] == 7:
                    #     annotationNew['category_id'] = 3 
                    # elif annotations[x]['category_id'] == 9:
                    #     annotationNew['category_id'] = 12
                    # elif annotations[x]['category_id'] == 17:
                    #     annotationNew['category_id'] = 1
                    # elif annotations[x]['category_id'] == 18:
                    #     annotationNew['category_id'] = 2
                    # elif annotations[x]['category_id'] == 19:
                    #     annotationNew['category_id'] = 5
                    # elif annotations[x]['category_id'] == 20:
                    #     annotationNew['category_id'] = 6
                    # elif annotations[x]['category_id'] == 21:
                    #     annotationNew['category_id'] = 8
                    
                    batchdata["annotations"].append(annotationNew)
            else: 
                annotaionsStartidnew = x
                break 

    # {"id":1,"name":"paragraph","supercategory":"document","color":"#7c37fb","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":2,"name":"title","supercategory":"document","color":"#12b5eb","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":3,"name":"table","supercategory":"document","color":"#b63019","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":4,"name":"list","supercategory":"document","color":"#adc70a","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":5,"name":"figure","supercategory":"document","color":"#65b92b","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":6,"name":"footer","supercategory":"document","color":"#36783d","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":7,"name":"formula","supercategory":"document","color":"#366b7d","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":8,"name":"page-header","supercategory":"document","color":"#0d06cb","metadata":{},"creator":"liangxuchao","keypoint_colors":[]},
    # {"id":9,"name":"document","supercategory":"","color":"#e73bcc","metadata":{},"creator":"liangxuchao","keypoint_colors":[]}
        # print(i)   
    # batchdata["categories"] = [{"supercategory": "", "id": 1, "name": "text"}, 
    # {"supercategory": "", "id": 2, "name": "title"}, 
    # {"supercategory": "", "id": 3, "name": "list"}, 
    # {"supercategory": "", "id": 4, "name": "table"}, 
    # {"supercategory": "", "id": 5, "name": "figure"},
    # {"supercategory": "", "id": 6, "name": "fomula"},
    # {"supercategory": "", "id": 7, "name": "document"}]

    batchdata["categories"] = [
    #   {
    #   "supercategory": "Caption",
    #   "id": 1,
    #   "name": "Caption"
    # },
    # {
    #   "supercategory": "Footnote",
    #   "id": 2,
    #   "name": "Footnote"
    # },
    # {
    #   "supercategory": "Formula",
    #   "id": 3,
    #   "name": "Formula"
    # },
    # {
    #   "supercategory": "List-item",
    #   "id": 4,
    #   "name": "List-item"
    # },
    # {
    #   "supercategory": "Page-footer",
    #   "id": 5,
    #   "name": "Page-footer"
    # },
    # {
    #   "supercategory": "Page-header",
    #   "id": 6,
    #   "name": "Page-header"
    # },
    # {
    #   "supercategory": "Picture",
    #   "id": 7,
    #   "name": "Picture"
    # },
    # {
    #   "supercategory": "Section-header",
    #   "id": 8,
    #   "name": "Section-header"
    # },
    # {
    #   "supercategory": "Table",
    #   "id": 9,
    #   "name": "Table"
    # },
    # {
    #   "supercategory": "Text",
    #   "id": 10,
    #   "name": "Text"
    # },
    # {
    #   "supercategory": "Title",
    #   "id": 11,
    #   "name": "Title"
    # },
    {"supercategory": "", "id": 12, "name": "document"}]
    train_json_string = JS.dumps(batchdata)
    with open(outputPath, 'w') as outfile:
        outfile.write(train_json_string)
    return annotaionsStartidnew

images = tuple(data['images'])
annotations = tuple(data['annotations'])

annotaionsStartid1 = changeAnnotationID(0,0,50,images,annotations,r'E:\fyp\warpdoc\withcoco\train_50.json')
# annotaionsStartid2 = changeAnnotationID(annotaionsStartid1,15,20,images,annotations,r'E:\fyp\warpdoc\withcoco\val_incomplete_5_1.json')
# annotaionsStartid2 = changeAnnotationID(annotaionsStartid1,60000,66000,images,annotations,r'E:\fyp\warpdoc\withcoco\val_4_1.json')

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

