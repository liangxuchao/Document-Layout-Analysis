import logging
import os
from collections import OrderedDict
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.data import build_detection_test_loader
register_coco_instances("PubLayNet_val", {}, "../datasets/publaynet_full/val.json", "../datasets/publaynet_full/val")
cfg = get_cfg()
cfg.merge_from_file('../configs/mask_rcnn_R_101_FPN_3x_Test.yaml')
cfg.DATASETS.TEST = ('PubLayNet_val')
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "../output/model_final.pth"  

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
evaluator = COCOEvaluator("PubLayNet_val", ("bbox", "segm"), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "PubLayNet_val")
predictor = DefaultPredictor(cfg) 
#print(inference_on_dataset(predictor.model, val_loader, evaluator))


import time
import os 
import cv2
import random as rd
import numpy as np
test_data_dir = '../datasets/publaynet_full/val_other'
# predictor = DefaultPredictor(cfg) 
files = os.listdir(test_data_dir)
cpu_device = torch.device("cpu")
textImg_metadata = MetadataCatalog.get("PubLayNet_val")

textImg_data = DatasetCatalog.get("PubLayNet_val")


for file in rd.choices(files, k = 17):
    print(os.path.join(test_data_dir, file))
    im = cv2.imread(os.path.join(test_data_dir, file))
    # im = cv2.imread('/content/242028536_25061232_page27.jpg')
    outputs = predictor(im)
    print("outputs")
    print(outputs)
    v = Visualizer(im[:, :, ::-1], textImg_metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # vis_output = visualizer.draw_panoptic_seg_predictions(
    #             panoptic_seg.to(cpu_device), segments_info
    #         )
    #v = v.cpu().numpy()
    img = v.get_image()[:, :, ::-1]

    out_file_name = file

    cv2.imwrite(out_file_name, img)
    # image = cv2.imread(img)

