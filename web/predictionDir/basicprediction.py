# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import cv2

import json as JS
import os 
import random as rd

coco = "E:/fyp/doclaynet/COCO/val_doclaynet.json"

test_data_dir = "E:/fyp/doclaynet/PNG/"
# Create config
cfg = get_cfg()   
cfg.merge_from_file('../configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = "../output/model_final.pth"  
cfg.MODEL.WEIGHTS = "E:/fyp/model/model_final_doclaynet_1460000.pth"
cfg.MODEL.DEVICE  = 'cpu'


instancename = 'val'
register_coco_instances(instancename, {}, coco, test_data_dir)
textImg_metadata = MetadataCatalog.get(instancename)

textImg_data = DatasetCatalog.get(instancename)

testimgDir = "C:/Users/Liang Xu Chao/Documents/ntu/project/predictionDir/testImg"
outputimgDir = "C:/Users/Liang Xu Chao/Documents/ntu/project/predictionDir/testImg/predictonOrigin/"


for filename in [f for f in os.listdir(testimgDir) if os.path.isfile(os.path.join(testimgDir, f))]:
    print(filename)
    im = cv2.imread(os.path.join(testimgDir, filename))
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Make prediction
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], textImg_metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    
    img = v.get_image()[:, :, ::-1]
    out_file_name = os.path.join(outputimgDir, filename)
    print(out_file_name)
    cv2.imwrite(out_file_name, img)
