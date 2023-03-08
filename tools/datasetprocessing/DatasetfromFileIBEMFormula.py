import json
import os
import cv2

classname_id = {'embedded': 1, 'isolated': 2}

class Math2Txt:
    def __init__(self, img_path, txt_path, dst_path):
        self.img_path = img_path
        self.txt_path = txt_path
        self.dst_path = dst_path
        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)

    def convert2Txt(self):
        for file in os.listdir(self.txt_path):
            img_path = os.path.join(self.img_path, file.replace('color_', '').replace('txt', 'jpg'))
            txt_path = os.path.join(self.txt_path, file)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            g = open(os.path.join(self.dst_path, file.replace('color_', '')), 'w')
            with open(txt_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if i < 4:
                        continue
                    line = line.replace(' ', '')
                    line = line.strip('\n').split('\t')
                    label = line[-1]
                    x1 = int(float(line[0]) * w / 100)
                    y1 = int(float(line[1]) * h / 100)
                    x2 = int(x1 + float(line[2]) * w / 100) - 1
                    y2 = y1
                    x3 = x2
                    y3 = int(y1 + float(line[3]) * h / 100) - 1
                    x4 = x1
                    y4 = y3
                    data = str(x1) + '\t' + str(y1) + '\t' + str(x2) + '\t' + str(y2) + '\t' \
                           + str(x3) + '\t' + str(y3) + '\t' + str(x4) + '\t' + str(y4) + '\t' + label + '\n'
                    print(data)
                    g.write(data)
                g.close()


class Txt2CoCo:
    def __init__(self, txt_path, img_path):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.txt_path = txt_path
        self.img_path = img_path

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)

    def convert2Coco(self):
        self._init_categories()
        print(self.txt_path)
        for im in os.listdir(self.img_path):
            im_name = im.split('.')[0]
            self.images.append(self._image(im))
            txt = im_name + '.txt'
            with open(os.path.join(self.txt_path, txt), 'r', encoding='utf-8') as f:
                for i, ann in enumerate(f.readlines()):
                    annotation, flag = self._annotation(ann)
                    if not flag:
                        continue
                    else:
                        self.annotations.append(annotation)
                        self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'COCO form created'
        instance['license'] = 'MIT'
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for k, v in classname_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    def _image(self, im_name):
        img = cv2.imread(os.path.join(img_path, im_name))
        try:
            H, W = img.shape[:-1]
        except:
            raise ValueError(im_name)
        image = {}
        image['height'] = H
        image['width'] = W
        image['id'] = self.img_id
        image['file_name'] = im_name
        return image

    def _annotation(self, ann):
        flag = False
        annotation = {}
        ann = ann.strip('\n').split('\t')
        label = int(ann[-1]) + 1
        annotation['category_id'] = label

        flag = True
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['segmentation'] = [self.str2int(ann[:-1])]
        annotation['bbox'] = self.segm2Bbox(annotation['segmentation'])
        annotation['iscrowd'] = 0
        annotation['area'] = self.getArea(annotation['bbox'])
        return annotation, flag

    def str2int(self, msg):
        coord = []
        for num in msg:
            coord.append(int(num))
        return coord

    def segm2Bbox(self, segm):
        x_min = min(segm[0][0::2])
        x_max = max(segm[0][0::2])
        y_min = min(segm[0][1::2])
        y_max = max(segm[0][1::2])
        bbox = [x_min, y_min, x_max-x_min+1, y_max-y_min+1]
        return bbox

    def getArea(self, bbox):
        return float(bbox[2] * bbox[3])

import json as JS

if __name__ == '__main__':
    # img_path = r'E:\fyp\ibem\val'
    # txt_path = r'E:\fyp\ibem\val_annotation'
    # dst_path = r'E:\fyp\ibem\txtlabel_val'
    # train_path = r'E:\fyp\ibem'

    # toIcdar = Math2Txt(img_path, txt_path, dst_path)
    # toIcdar.convert2Txt()

    # toCoco = Txt2CoCo(dst_path, img_path)
    # instance = toCoco.convert2Coco()
    # toCoco.save_coco_json(instance, os.path.join(train_path, 'val.json'))

    # fullannotaPath = r'E:\fyp\ibem\train.json'
    # outputPath = r'E:\fyp\ibem\train_11category.json'
    fullannotaPath = r'E:\fyp\ibem\train.json'
    outputPath = r'E:\fyp\ibem\train_12category.json'
    with open(fullannotaPath, "r") as json_file:
        data = JS.load(json_file)
    annotations = tuple(data['annotations'])
    batchdata = {"images":[],"annotations":[],"categories":[]}

    for x in range(0,len(annotations)):
        
        if annotations[x]['category_id'] in [2]:
            annotationNew = annotations[x]
            
            if annotations[x]['category_id'] ==2:
                annotationNew['category_id'] = 3
            batchdata["annotations"].append(annotationNew)
    
    batchdata["images"] = data['images']
        # print(i)   
    # batchdata["categories"] = [{"supercategory": "", "id": 1, "name": "text"}, 
    # {"supercategory": "", "id": 2, "name": "title"}, 
    # {"supercategory": "", "id": 3, "name": "list"}, 
    # {"supercategory": "", "id": 4, "name": "table"}, 
    # {"supercategory": "", "id": 5, "name": "figure"},
    # {"supercategory": "", "id": 6, "name": "fomula"},
    # {"supercategory": "", "id": 7, "name": "document"}]
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

# import os
# import shutil
# import os
# opath = r"E:\fyp\ibem\Ts11"
# for file in os.listdir(opath):
#     if file.endswith(".txt"):
#         shutil.move(os.path.join(opath, file), os.path.join(r"E:\fyp\ibem\train_annotation", file))
#     elif file.endswith(".jpg"):
#         shutil.move(os.path.join(opath, file), os.path.join(r"E:\fyp\ibem\train", file))
# # shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")