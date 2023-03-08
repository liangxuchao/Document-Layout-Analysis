# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog,build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
import cv2
import tensorflow as tf
import json as JS
import os 
import random as rd
from PIL import Image
import numpy as np
from detectron2.modeling import GeneralizedRCNNWithTTA
from collections import OrderedDict
from detectron2.modeling import build_model
import torch
class documentPreditor:
    def __init__(self,dmodel,dconfig,lmodel,lconfig,dclevel,lclevel,cocometajson1,cocometajson2,outputdocumentDir,outputimgDir,outputdigitalimgDir):

        self.cfg = get_cfg()
        self.cfg.merge_from_file(dconfig)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = dclevel # set threshold for this model
        self.cfg.MODEL.WEIGHTS = dmodel 

        self.cfg2 = get_cfg()
        self.cfg2.merge_from_file(lconfig)
        self.cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = lclevel # set threshold for this model
        self.cfg2.MODEL.WEIGHTS = lmodel

        self.dpredictor = DefaultPredictor(self.cfg)

        # self.lpredictor = Predictor.DefaultPredictor(self.cfg2,True)
        self.lpredictor = DefaultPredictor(self.cfg2)
        self.documentfile = []
        self.outputdocumentDir = outputdocumentDir
        self.outputimgDir = outputimgDir
        self.outputdigitalimgDir = outputdigitalimgDir

        DatasetCatalog.clear()

        instancename = 'document'
        register_coco_instances(instancename, {}, cocometajson1, "")
        self.textImg_metadata = MetadataCatalog.get(instancename)

        MetadataCatalog.get(instancename).thing_classes = [ "Document"]

        
        instancename = 'documentContent'
        register_coco_instances(instancename, {}, cocometajson2, "")
        self.textImg_metadata = MetadataCatalog.get(instancename)
        MetadataCatalog.get(instancename).thing_classes = [ "Caption", "Footnote", "Formula","List-item","Page-footer", 
        "Page-header",  "Picture","Section-header",  "Table", "Text","Title","Document"]


    def predictImages(self,imgpath):
        files = [f for f in os.listdir(imgpath) if os.path.isfile(os.path.join(imgpath, f))]
        for filename in files:
            print(filename)
            fname, file_extension = os.path.splitext(filename)
            img = cv2.imread(os.path.join(imgpath, filename))

            documentbox = self.predictDocument(img)
            # print(img.shape)
            marginHeight = int(img.shape[0] * 0.01)
            marginWidth = int(img.shape[1] * 0.01)
            # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

           
            cropInfo = {"image":[],"indexs":[],"pred":[]}
            if len(documentbox) == 0:
                self.documentfile.append(os.path.join(imgpath, filename))

            for i in range(0,len(documentbox)):
                [X, Y, W, H] = documentbox[i]
                if Y - marginHeight > 0:
                    Y = Y- marginHeight

                if H + marginHeight < img.shape[1]:
                    H = H + marginHeight
                
                if X - marginWidth > 0:
                    X = X- marginWidth

                if W + marginWidth < img.shape[1]:
                    W = W + marginWidth
                
                cropped_image = img[int(Y):int(H), int(X):int(W)]
                cropInfo["indexs"].append([int(X),int(Y),int(W),int(H)])
                out_file_name = os.path.join(self.outputdocumentDir,fname + "_" + str(i) + file_extension)
                cropInfo["image"].append(cropped_image)
                cv2.imwrite(out_file_name, cropped_image)
                self.documentfile.append(out_file_name)
                
              

            # out_file_name =os.path.join(self.outputimgDir, filename)  
            # self.applyPredOnOrigin(img,cropInfo,out_file_name)
        for d in self.documentfile:
            im = cv2.imread(d)
            # Make prediction
            
            outputs = self.lpredictor(im)
            v = Visualizer(im[:, :, ::-1], self.textImg_metadata, scale=1)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
              # Get the predicted classes and confidence scores
            # scores = outputs["instances"].scores.to("cpu").numpy()
            # classes = outputs["instances"].pred_classes.to("cpu").numpy()
            # boxes = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()

            # indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.7)
            # v = Visualizer(im[:, :, ::-1], self.textImg_metadata, scale=1)
            # v = v.draw_instance_predictions(outputs["instances"][indices].to("cpu"))
            
            
            img = v.get_image()[:, :, ::-1]
            out_file_name =os.path.join(self.outputimgDir, os.path.basename(d))
            print(out_file_name)
            cv2.imwrite(out_file_name, img)
        return

    def predictDocument(self,img):

        # Make prediction
        outputs = self.dpredictor(img)

        # Create predictor
        classes = outputs["instances"].pred_classes.to("cpu")
        boxes = outputs["instances"].pred_boxes.to("cpu")

        documents = []
        for i in range(0,len(classes)):
            
            if classes[i] == 0:
                documents.append(boxes[i].tensor.numpy()[0])
        print(documents)
        return  documents

    def test_with_TTA(cls, cfg, model):
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        model = GeneralizedRCNNWithTTA(cfg, model)
        
        return model

    def simplePredict(self,imgDir):
       for filename in [f for f in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, f))]:
            print(filename)
            im = cv2.imread(os.path.join(imgDir, filename))
            # Create predictor
            outputs = self.lpredictor(im)
            

            v = Visualizer(im[:, :, ::-1], self.textImg_metadata, scale=1)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
            img = v.get_image()[:, :, ::-1]
            out_file_name = os.path.join(self.outputdigitalimgDir, filename)
            print(out_file_name)
            cv2.imwrite(out_file_name, img)


if __name__ == "__main__":
    with open('setting.json') as user_file:
        file_contents = user_file.read()
    
    parsed_json = JS.loads(file_contents)
    predictor = documentPreditor(parsed_json["PredictDocumentModel"],
                                 parsed_json["ConfigFilePreidctOnDocument"],
                                 parsed_json["PredictLayoutModel"],
                                 parsed_json["ConfigFilePreidctOnLayout"],
                                 parsed_json["PredDocumentThreshold"],
                                 parsed_json["PredLayoutThreshold"],
                                 parsed_json["COCOJsonDocument"],
                                 parsed_json["COCOJsonLayout"],
                                 parsed_json["OutPutDocumentDir"],
                                 parsed_json["OutPutLayoutDir"],
                                 parsed_json["OutPutOriginDir"],
                                 )
    predictor.predictImages(parsed_json["TestImgDir"])