import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as mpatches
import numpy as np
import json as JS
import imgtools

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

# annotation_path = r"E:\fyp\warpdoc\withcoco\train_incomplete_15_1.json"
# image_dir = r'E:\fyp\warpdoc\WarpDoc\image\incomplete'
# theta = 10
# bb1 = {}
# x1=551.6409313725491 
# x2=551.6409313725491+108.7790126633987
# y1=131.09775883838392
# y2=131.09775883838392+7.338197601010052

# bb1[0] = [(float(x1),float(y1)), (float(x2),float(y1)),
# (float(x1),float(y2)),(float(x2),float(y2))]

# print(bb1)

# # Original image
# img_orig = cv2.imread(r'E:\fyp\doclaynet\PNG\d4bbc62a06f5b9ed82c9e101f668858ce6e30c6f148991d229a45fd68228ccf7.png')
# # Rotated image
# rotated_img = imgtools.rotate_bound(img_orig, theta)
# # Calculate the shape of rotated images

def CalculateMatrix(img_orig,rotated_img):
        
    (heigth, width) = img_orig.shape[:2]
    (cx, cy) = (width // 2, heigth // 2)
    (new_height, new_width) = rotated_img.shape[:2]
    (new_cx, new_cy) = (new_width // 2, new_height // 2)
    return cx,cy,heigth,width

# ## Calculate the new bounding box coordinates
# new_bb = {}
# for i in bb1:
#     new_bb[i] = imgtools.rotate_box(bb1[i], cx, cy, heigth, width)
#     print(new_bb[i])

# name1=r'C:\Users\Liang Xu Chao\Documents\ntu\project\tools\testImg\Output.png'
# name2=r'C:\Users\Liang Xu Chao\Documents\ntu\project\tools\testImg\Output2.png'
# cv2.imwrite(name1, img_orig)
# cv2.imwrite(name2, rotated_img)


def rotation(annotaionsStartid,startImg,endImg,images,annotations,outputPathAnno,outputPathimg,imgfolder,categories,theta):
    annotaionsStartidnew = annotaionsStartid
    print(annotaionsStartidnew)
    batchdata = {"images":[],"annotations":[],"categories":[]}
    
    for i in range(startImg,endImg):
        imageadd = False
        img_orig = cv2.imread(imgfolder + images[i]['file_name'])
        rotated_img = imgtools.rotate_bound(img_orig, theta)
        rotateMatrix = CalculateMatrix(img_orig,rotated_img)
        for x in range(annotaionsStartidnew,len(annotations)):
            if annotations[x]['image_id'] == images[i]['id']:
              if annotations[x]['category_id'] == 12:
                # annotationNew = annotations[x]
                x1=annotations[x]['bbox'][0]
                x2=annotations[x]['bbox'][0]+annotations[x]['bbox'][2]
                y1=annotations[x]['bbox'][1]
                y2=annotations[x]['bbox'][1]+annotations[x]['bbox'][3]
                segmentation = annotations[x]['segmentation'][0]
                bb = [(float(x1),float(y1)), (float(x2),float(y1)),
                (float(x1),float(y2)),(float(x2),float(y2))]
                newseg = []
                for s in range(0,len(segmentation),2):
                  newseg.append((segmentation[s],segmentation[s+1]))
                new_seg = imgtools.rotate_box(newseg, rotateMatrix[0], rotateMatrix[1], rotateMatrix[2], rotateMatrix[3],theta)
                new_bb = imgtools.rotate_box(bb, rotateMatrix[0], rotateMatrix[1], rotateMatrix[2], rotateMatrix[3],theta)

                minx = new_bb[0][0]
                miny = new_bb[0][1]
                maxx = new_bb[0][0]
                maxy = new_bb[0][1]
                for z in range(1,len(new_bb)):
                  if new_bb[z][0] < minx:
                    minx = new_bb[z][0]
                  elif new_bb[z][0] > maxx:
                    maxx = new_bb[z][0]
                    
                  if new_bb[z][1] < miny:
                    miny = new_bb[z][1]
                  elif new_bb[z][1] > maxy:
                    maxy = new_bb[z][1]

                annotations[x]['bbox'][0] = int(minx)
                annotations[x]['bbox'][1] = int(miny)
                annotations[x]['bbox'][2] = int(maxx -minx)
                annotations[x]['bbox'][3] = int(maxy -miny)

                annotations[x]['segmentation'][0] = []
                for ns in range(0,len(new_seg)):
                  annotations[x]['segmentation'][0].append(new_seg[ns][0])
                  annotations[x]['segmentation'][0].append(new_seg[ns][1])
               
                # annotations[x]['segmentation'][0][0] = new_seg[0][0]
                # annotations[x]['segmentation'][0][1] = new_seg[0][1]
                # annotations[x]['segmentation'][0][2] = new_seg[1][0] 
                # annotations[x]['segmentation'][0][3] = new_seg[1][1] 
                # annotations[x]['segmentation'][0][4] = new_seg[2][0]
                # annotations[x]['segmentation'][0][5] = new_seg[2][1]
                # annotations[x]['segmentation'][0][6] = new_seg[3][0] 
                # annotations[x]['segmentation'][0][7] = new_seg[3][1] 
                # annotations[x]['segmentation'] = [[]]

                imageadd = True
                batchdata["annotations"].append(annotations[x])

            else: 
                annotaionsStartidnew = x
                break 
        if imageadd:
            
            images[i]['width'] = rotated_img.shape[1]
            images[i]['height'] = rotated_img.shape[0]
            # cv2.imwrite(outputPathimg + images[i]['file_name'], rotated_img)
            batchdata["images"].append(images[i])
    
    batchdata["categories"] = categories
    train_json_string = JS.dumps(batchdata)
    with open(outputPathAnno, 'w') as outfile:
        outfile.write(train_json_string)
   
    return annotaionsStartidnew

#fullannotaPath =  r"E:\fyp\warpdoc\withcoco\val_wrap.json"
fullannotaPath =  r"E:\fyp\doclaynet_rotate\COCO\train_69375_rotate15.json"


with open(fullannotaPath, "r") as json_file:
    data = JS.load(json_file)

count = 0
# from PIL import Image
# def paddingTransformation():
      
#   image = Image.open("input.jpg")
    
#   right = 100
#   left = 100
#   top = 100
#   bottom = 100
    
#   width, height = image.size
    
#   new_width = width + right + left
#   new_height = height + top + bottom
    
#   result = Image.new(image.mode, (new_width, new_height), (0, 0, 255))
    
#   result.paste(image, (left, top))
    
#   result.save('output.jpg')


categories = [
  # {
  #     "supercategory": "Caption",
  #     "id": 1,
  #     "name": "Caption"
  #   },
  #   {
  #     "supercategory": "Footnote",
  #     "id": 2,
  #     "name": "Footnote"
  #   },
  #   {
  #     "supercategory": "Formula",
  #     "id": 3,
  #     "name": "Formula"
  #   },
  #   {
  #     "supercategory": "List-item",
  #     "id": 4,
  #     "name": "List-item"
  #   },
  #   {
  #     "supercategory": "Page-footer",
  #     "id": 5,
  #     "name": "Page-footer"
  #   },
  #   {
  #     "supercategory": "Page-header",
  #     "id": 6,
  #     "name": "Page-header"
  #   },
  #   {
  #     "supercategory": "Picture",
  #     "id": 7,
  #     "name": "Picture"
  #   },
  #   {
  #     "supercategory": "Section-header",
  #     "id": 8,
  #     "name": "Section-header"
  #   },
  #   {
  #     "supercategory": "Table",
  #     "id": 9,
  #     "name": "Table"
  #   },
  #   {
  #     "supercategory": "Text",
  #     "id": 10,
  #     "name": "Text"
  #   },
  #   {
  #     "supercategory": "Title",
  #     "id": 11,
  #     "name": "Title"
  #   },
    {"supercategory": "", "id": 12, "name": "document"}]

images = tuple(data['images'])
annotations = tuple(data['annotations'])
# annotaionsStartid1 = rotation(0,0,5,images,annotations,'E:/fyp/doclaynet_rotate/COCO/val_5_rotate15.json',
# 'E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,15)


# annotaionsStartid1 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotate15.json','E:/fyp/doclaynet_rotate/PNG15/','E:/fyp/doclaynet/PNG/',categories,15)
# annotaionsStartid2 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotate30.json','E:/fyp/doclaynet_rotate/PNG30/','E:/fyp/doclaynet/PNG/',categories,30)
# annotaionsStartid3 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotate45.json','E:/fyp/doclaynet_rotate/PNG45/','E:/fyp/doclaynet/PNG/',categories,45)
# annotaionsStartid4 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotate60.json','E:/fyp/doclaynet_rotate/PNG60/','E:/fyp/doclaynet/PNG/',categories,60)
# annotaionsStartid5 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotate75.json','E:/fyp/doclaynet_rotate/PNG75/','E:/fyp/doclaynet/PNG/',categories,75)
# annotaionsStartid6 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotate90.json','E:/fyp/doclaynet_rotate/PNG90/','E:/fyp/doclaynet/PNG/',categories,90)
# annotaionsStartid7 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotateN15.json','E:/fyp/doclaynet_rotate/PNGN15/','E:/fyp/doclaynet/PNG/',categories,-15)
# annotaionsStartid8 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotateN30.json','E:/fyp/doclaynet_rotate/PNGN30/','E:/fyp/doclaynet/PNG/',categories,-30)
# annotaionsStartid9 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotateN45.json','E:/fyp/doclaynet_rotate/PNGN45/','E:/fyp/doclaynet/PNG/',categories,-45)
# annotaionsStartid10 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_69375_rotateN60.json','E:/fyp/doclaynet_rotate/PNGN60/','E:/fyp/doclaynet/PNG/',categories,-60)
# annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\warpdoc\withcoco\train_50_rotateN75_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/N75/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,-75)

# annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\warpdoc\withcoco\train_50_rotateN45_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/N45/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,-45)
# annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\warpdoc\withcoco\train_50_rotateN15_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/N15/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,-15)
# annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\warpdoc\withcoco\train_50_rotate75_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/75/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,75)
# annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\warpdoc\withcoco\train_50_rotate45_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/45/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,45)
# annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\warpdoc\withcoco\train_50_rotate15_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/15/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,15)

annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_rotate15_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/N75/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,-75)

annotaionsStartid11 = rotation(0,0,len(images),images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_rotateN15_1c.json','E:/fyp/warpdoc/WarpDoc/image/transformation/N45/','E:/fyp/warpdoc/WarpDoc/image/rotate/',categories,-45)

# annotaionsStartid1 = rotation(0,0,6300,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotate15.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,15)
# annotaionsStartid2 = rotation(annotaionsStartid1,6300,12600,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotate30.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,30)
# annotaionsStartid3 = rotation(annotaionsStartid2,12600,18900,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotate45.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,45)
# annotaionsStartid4 = rotation(annotaionsStartid3,18900,25200,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotate60.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,60)
# annotaionsStartid5 = rotation(annotaionsStartid4,25200,31500,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotate75.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,75)
# annotaionsStartid6 = rotation(annotaionsStartid5,31500,37800,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotate90.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,90)
# annotaionsStartid7 = rotation(annotaionsStartid6,37800,44100,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotateN15.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-15)
# annotaionsStartid8 = rotation(annotaionsStartid7,44100,50400,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotateN30.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-30)
# annotaionsStartid9 = rotation(annotaionsStartid8,50400,56700,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotateN45.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-45)
# annotaionsStartid10 = rotation(annotaionsStartid9,56700,63000,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotateN60.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-60)
# annotaionsStartid11 = rotation(annotaionsStartid10,63000,69000,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\train_6300_rotateN75.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-75)



# annotaionsStartid1 = rotation(0,0,500,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotate15.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,15)
# annotaionsStartid2 = rotation(annotaionsStartid1,500,1000,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotate30.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,30)
# annotaionsStartid3 = rotation(annotaionsStartid2,1000,1500,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotate45.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,45)
# annotaionsStartid4 = rotation(annotaionsStartid3,1500,2000,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotate60.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,60)
# annotaionsStartid5 = rotation(annotaionsStartid4,2000,2500,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotate75.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,75)
# annotaionsStartid6 = rotation(annotaionsStartid5,2500,3000,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotate90.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,90)
# annotaionsStartid7 = rotation(annotaionsStartid6,3000,3500,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotateN15.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-15)
# annotaionsStartid8 = rotation(annotaionsStartid7,3500,4000,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotateN30.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-30)
# annotaionsStartid9 = rotation(annotaionsStartid8,4000,4500,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotateN45.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-45)
# annotaionsStartid10 = rotation(annotaionsStartid9,4500,5000,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotateN60.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-60)
# annotaionsStartid11 = rotation(annotaionsStartid10,5000,5500,images,annotations,r'E:\fyp\doclaynet_rotate\COCO\val_500_rotateN75.json','E:/fyp/doclaynet_rotate/PNG/','E:/fyp/doclaynet/PNG/',categories,-75)