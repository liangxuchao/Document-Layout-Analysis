import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import torch
import detectron2
from tqdm.auto import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import DatasetCatalog, build_detection_test_loader
import pycocotools.mask as mask_util
from PIL import Image
import matplotlib.pyplot as plt
from fastcore.all import *
from ensemble_boxes import *


from PIL import Image as im
class ensemblePredictor():
    def __init__(self, configs):
        self.MIN_PIXELS = configs["min_pixels"]
        self.IOU_TH = configs["iou_th"]
        self.MODELS = []
        self.THSS = []
        self.ID_TEST = 0

        for b_m in configs["models"]:
            model_name = b_m["file"]
            model_ths = b_m["ths"]
            config_name = b_m["config_name"]
            self.THSS.append(model_ths)
            cfg = get_cfg()
            cfg.merge_from_file(config_name)
            cfg.INPUT.MASK_FORMAT = 'bitmask'
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = configs["number_classes"]
            cfg.MODEL.WEIGHTS = model_name
            cfg.TEST.DETECTIONS_PER_IMAGE = configs["detection_per_img"]
            self.MODELS.append(DefaultPredictor(cfg))

    def predict(self, imagePath):

        img = cv2.imread(imagePath)
        subm_ids, subm_masks = [], []
        encoded_masks_single = self.pred_masks(
            img,
            model=self.MODELS[0],
            ths=self.THSS[0],
            min_pixels=self.MIN_PIXELS
        )

        classes, scores, bboxes, masks = self.ensemble_preds(
            file_name=imagePath,
            models=self.MODELS,
            ths=self.THSS
        )
        # if NMS:
        #     classes, scores, masks = nms_predictions(
        #         classes,
        #         scores,
        #         bboxes,
        #         masks, iou_th=IOU_TH
        #     )
        encoded_masks = self.ensemble_pred_masks(
            masks, classes, self.MIN_PIXELS, img.shape[:2])

        for enc_mask in encoded_masks:
            subm_ids.append(imagePath[:imagePath.find('.')])
            subm_masks.append(enc_mask)
        # _, axs = plt.subplots(2, 2, figsize=(14, 8))
        # axs[0][0].imshow(cv2.imread(imagePath))
        # axs[0][0].axis('off')
        # axs[0][0].set_title("image")
        # for en_mask in encoded_masks_single:
        #     dec_mask = self.rle_decode(en_mask, img.shape[:2])
        #     axs[0][1].imshow(img)
        #     axs[0][1].imshow(np.ma.masked_where(dec_mask == 0, dec_mask))
        #     axs[0][1].axis('off')
        #     axs[0][1].set_title('single model')
        # axs[1][0].imshow(cv2.imread(imagePath))
        # axs[1][0].axis('off')
        # axs[1][0].set_title('image')
        # for en_mask in encoded_masks:
        #     dec_mask = self.rle_decode(en_mask, img.shape[:2])
        #     axs[1][1].imshow(img,alpha=0.3, interpolation='none')
        #     axs[1][1].imshow(np.ma.masked_where(dec_mask == 0, dec_mask), interpolation='none', alpha=0.7)
           
        #     axs[1][1].axis('off')
        #     axs[1][1].set_title('ensemble models')
        # plt.show()

        
        dec_mask = self.rle_decode(encoded_masks[0], img.shape[:2]) 
        masked = cv2.bitwise_and(img,img, mask=dec_mask)
        for en_mask in range(1,len(encoded_masks)):
            dec_mask = self.rle_decode(encoded_masks[en_mask], img.shape[:2]) 
            masked = cv2.bitwise_and(masked,masked,mask= dec_mask)

        
        cv2.imshow("Mask Applied to Image", masked)
        cv2.waitKey(0)
    def rle_decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background

        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int)
                           for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo: hi] = 1
        return img.reshape(shape)  # Needed to align to RLE direction

    def rle_encode(self, img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        
        '''
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def pred_masks(self, img, model, ths, min_pixels):
        output = model(img)
        pred_classes = output['instances'].pred_classes.cpu().numpy().tolist()
        pred_class = max(set(pred_classes), key=pred_classes.count)
        take = output['instances'].scores >= ths[pred_class]
        pred_masks = output['instances'].pred_masks[take]
        pred_masks = pred_masks.cpu().numpy()
        result = []
        used = np.zeros(img.shape[:2], dtype=int)
        for i, mask in enumerate(pred_masks):
            mask = mask * (1 - used)
            if mask.sum() >= min_pixels[pred_class]:
                used += mask
                result.append(self.rle_encode(mask))
        return result

    def ensemble_preds(self, file_name, models, ths):
        img = cv2.imread(file_name)
        classes = []
        scores = []
        bboxes = []
        masks = []
        for i, model in enumerate(models):
            output = model(img)
            pred_classes = output['instances'].pred_classes.cpu(
            ).numpy().tolist()
            pred_class = max(set(pred_classes), key=pred_classes.count)
            take = output['instances'].scores >= ths[i][pred_class]
            classes.extend(
                output['instances'].pred_classes[take].cpu().numpy().tolist())
            scores.extend(
                output['instances'].scores[take].cpu().numpy().tolist())
            bboxes.extend(
                output['instances'].pred_boxes[take].tensor.cpu().numpy().tolist())
            masks.extend(output['instances'].pred_masks[take].cpu().numpy())
        assert len(classes) == len(masks), 'ensemble lenght mismatch'
        #scores, classes, bboxes, masks = zip(*sorted(zip(scores, classes, bboxes, masks),reverse=True))
        return classes, scores, bboxes, masks

    # def nms_predictions(self,classes, scores, bboxes, masks,
    #                     iou_th, shape):

    #     print(bboxes)
    #     print(masks)
    #     print(classes)
    #     he, wd = shape[0], shape[1]
    #     boxes_list = [[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he]
    #                 for x in bboxes]
    #     scores_list = [x for x in scores]
    #     labels_list = [x for x in classes]
    #     nms_bboxes, nms_scores, nms_classes = nms(
    #         boxes=[boxes_list],
    #         scores=[scores_list],
    #         labels=[labels_list],
    #         weights=None,
    #         iou_thr=iou_th
    #     )
    #     nms_masks = []
    #     for s in nms_scores:
    #         nms_masks.append(masks[scores.index(s)])
    #     nms_scores, nms_classes, nms_masks = zip(*sorted(zip(nms_scores, nms_classes, nms_masks), reverse=True))
    #     return nms_classes, nms_scores, nms_masks

    def ensemble_pred_masks(self, masks, classes, min_pixels, shape):
        result = []
        pred_class = max(set(classes), key=classes.count)
        used = np.zeros(shape, dtype=int)
        for i, mask in enumerate(masks):
            mask = mask * (1 - used)
            if mask.sum() >= min_pixels[pred_class]:
                used += mask
                result.append(self.rle_encode(mask))
        return result


if __name__ == "__main__":
    config = {"models": ({'file': 'E:/fyp/model/model_final_doclaynet_500000.pth', 
                          'config_name': '../configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml', 
                          'ths': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]},
                         {'file': 'E:/fyp/model/model_final_doclaynet_700000.pth', 
                          'config_name': '../configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml', 
                          'ths': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]}), 
              "iou_th": .5,
              "number_classes":11,
              "min_pixels": [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
              "detection_per_img":1000}
    ensembleModels = ensemblePredictor(config)
    imagePath = r"C:\Users\Liang Xu Chao\Documents\ntu\project\tools\testimg\7.jpg"
    ensembleModels.predict(imagePath)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
#     print('GPU is available')
# else:
#     DEVICE = torch.device('cpu')
#     print('CPU is used')
# print('detectron ver:', detectron2.__version__)


# best_model=( {'file': 'model_final_doclaynet_600000.pth','config_name':'../configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml','ths':[.5, .5, .5,.5, .5, .5,.5, .5, .5,.5, .5]},
#              {'file': 'model_final_doclaynet_700000.pth','config_name':'../configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml','ths':[.5, .5, .5,.5, .5, .5,.5, .5, .5,.5, .5]})

# mdl_path = "E:/fyp/model"
# DATA_PATH = 'C:/Users/Liang Xu Chao/Documents/ntu/project/tools/testImg/document/'
# MODELS = []
# BEST_MODELS =[]
# THSS = []
# ID_TEST = 0
# SUBM_PATH = f'{DATA_PATH}'
# SINGLE_MODE = False
# NMS = False
# MIN_PIXELS = [640, 640, 640, 640, 640, 640, 640, 640, 640, 640, 640,]
# IOU_TH = .5
# for b_m in best_model:
#     model_name=b_m["file"]
#     model_ths=b_m["ths"]
#     config_name=b_m["config_name"]
#     BEST_MODELS.append(model_name)
#     THSS.append(model_ths)
#     cfg = get_cfg()
#     cfg.merge_from_file(config_name)
#     cfg.INPUT.MASK_FORMAT = 'bitmask'
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
#     cfg.MODEL.WEIGHTS = f'{mdl_path}/{model_name}'
#     cfg.TEST.DETECTIONS_PER_IMAGE = 1000
#     MODELS.append(DefaultPredictor(cfg))
# print(f'all loaded:\nthresholds: {THSS}\nmodels: {BEST_MODELS}')


# def rle_decode(mask_rle, shape=(3283,2773)):
#     '''
#     mask_rle: run-length as string formated (start length)
#     shape: (height,width) of array to return
#     Returns numpy array, 1 - mask, 0 - background

#     '''
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int)
#                        for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo : hi] = 1
#     return img.reshape(shape)  # Needed to align to RLE direction

# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated

#     '''
#     pixels = img.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)

# def pred_masks(file_name, path, model, ths, min_pixels):
#     img = cv2.imread(f'{path}/{file_name}')
#     output = model(img)
#     pred_classes = output['instances'].pred_classes.cpu().numpy().tolist()
#     pred_class = max(set(pred_classes), key=pred_classes.count)
#     take = output['instances'].scores >= ths[pred_class]
#     pred_masks = output['instances'].pred_masks[take]
#     pred_masks = pred_masks.cpu().numpy()
#     result = []
#     used = np.zeros(img.shape[:2], dtype=int)
#     for i, mask in enumerate(pred_masks):
#         mask = mask * (1 - used)
#         if mask.sum() >= min_pixels[pred_class]:
#             used += mask
#             result.append(rle_encode(mask))
#     return result

# def ensemble_preds(file_name, path, models, ths):
#     img = cv2.imread(f'{path}/{file_name}')
#     classes = []
#     scores = []
#     bboxes = []
#     masks = []
#     for i, model in enumerate(models):
#         output = model(img)
#         pred_classes = output['instances'].pred_classes.cpu().numpy().tolist()
#         pred_class = max(set(pred_classes), key=pred_classes.count)
#         take = output['instances'].scores >= ths[i][pred_class]
#         classes.extend(output['instances'].pred_classes[take].cpu().numpy().tolist())
#         scores.extend(output['instances'].scores[take].cpu().numpy().tolist())
#         bboxes.extend(output['instances'].pred_boxes[take].tensor.cpu().numpy().tolist())
#         masks.extend(output['instances'].pred_masks[take].cpu().numpy())
#     assert len(classes) == len(masks) , 'ensemble lenght mismatch'
#     #scores, classes, bboxes, masks = zip(*sorted(zip(scores, classes, bboxes, masks),reverse=True))
#     return classes, scores, bboxes, masks

# def nms_predictions(classes, scores, bboxes, masks,
#                     iou_th=.5, shape=(3283,2773)):

#     print(bboxes)
#     print(masks)
#     print(classes)
#     he, wd = shape[0], shape[1]
#     boxes_list = [[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he]
#                   for x in bboxes]
#     scores_list = [x for x in scores]
#     labels_list = [x for x in classes]
#     nms_bboxes, nms_scores, nms_classes = nms(
#         boxes=[boxes_list],
#         scores=[scores_list],
#         labels=[labels_list],
#         weights=None,
#         iou_thr=iou_th
#     )
#     nms_masks = []
#     for s in nms_scores:
#         nms_masks.append(masks[scores.index(s)])
#     nms_scores, nms_classes, nms_masks = zip(*sorted(zip(nms_scores, nms_classes, nms_masks), reverse=True))
#     return nms_classes, nms_scores, nms_masks

# def ensemble_pred_masks(masks, classes, min_pixels, shape=(3283,2773)):
#     result = []
#     pred_class = max(set(classes), key=classes.count)
#     used = np.zeros(shape, dtype=int)
#     for i, mask in enumerate(masks):
#         mask = mask * (1 - used)
#         if mask.sum() >= min_pixels[pred_class]:
#             used += mask
#             result.append(rle_encode(mask))
#     return result

# test_names = os.listdir(SUBM_PATH)
# print('test images:', len(test_names))


# encoded_masks_single = pred_masks(
#     test_names[ID_TEST],
#     path=SUBM_PATH,
#     model=MODELS[0],
#     ths=THSS[0],
#     min_pixels=MIN_PIXELS
# )

# classes, scores, bboxes, masks = ensemble_preds(
#     file_name=test_names[ID_TEST] ,
#     path=SUBM_PATH,
#     models=MODELS,
#     ths=THSS
# )
# if NMS:
#     classes, scores, masks = nms_predictions(
#         classes,
#         scores,
#         bboxes,
#         masks, iou_th=IOU_TH
#     )
# encoded_masks = ensemble_pred_masks(masks, classes, min_pixels=MIN_PIXELS)


# _, axs = plt.subplots(2, 2, figsize=(14, 8))
# axs[0][0].imshow(cv2.imread(f'{SUBM_PATH}/{test_names[ID_TEST]}'))
# axs[0][0].axis('off')
# axs[0][0].set_title(test_names[ID_TEST])
# for en_mask in encoded_masks_single:
#     dec_mask = rle_decode(en_mask)
#     axs[0][1].imshow(np.ma.masked_where(dec_mask == 0, dec_mask))
#     axs[0][1].axis('off')
#     axs[0][1].set_title('single model')
# axs[1][0].imshow(cv2.imread(f'{SUBM_PATH}/{test_names[ID_TEST]}'))
# axs[1][0].axis('off')
# axs[1][0].set_title(test_names[ID_TEST])
# for en_mask in encoded_masks:
#     dec_mask = rle_decode(en_mask)
#     axs[1][1].imshow(np.ma.masked_where(dec_mask == 0, dec_mask))
#     axs[1][1].axis('off')
#     axs[1][1].set_title('ensemble models')
# plt.show()


# subm_ids, subm_masks = [], []
# for test_name in tqdm(test_names):
#     if SINGLE_MODE:
#         encoded_masks = pred_masks(
#             test_name,
#             path=SUBM_PATH,
#             model=MODELS[0],
#             ths=THSS[0],
#             min_pixels=MIN_PIXELS
#         )
#     else:
#         classes, scores, bboxes, masks = ensemble_preds(
#             file_name=test_name,
#             path=SUBM_PATH,
#             models=MODELS,
#             ths=THSS
#         )
#         if NMS:
#             classes, scores, masks = nms_predictions(
#                 classes,
#                 scores,
#                 bboxes,
#                 masks,
#                 iou_th=IOU_TH
#             )
#         encoded_masks = ensemble_pred_masks(
#             masks,
#             classes,
#             min_pixels=MIN_PIXELS
#         )
#     for enc_mask in encoded_masks:
#         subm_ids.append(test_name[:test_name.find('.')])
#         subm_masks.append(enc_mask)
