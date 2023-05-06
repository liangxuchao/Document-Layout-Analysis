# Document-Layout-Analysis

## Installation
```
$   pip install -r requirements.txt
```
### Install detectron2
Requirment
- CUDA=10.1 
- PyTorch>=3.7.0

How to install CUDA 10.1 can be found here: https://developer.nvidia.com/cuda-10.1-download-archive-base

How to install PyTorch can be found here: https://pytorch.org/

Afer installed above package, follow the instructions below to install detectron2:
```
$   git clone https://github.com/facebookresearch/detectron2.git
$   git checkout 8e3effc
$   python -m pip install -e detectron2
```


### Dataset

[IBM DocLayNet](https://developer.ibm.com/exchanges/data/all/doclaynet/) dataset for training and testing.

### Model

Model is under /output folder

model_final_document_80000.pth
model_final_doclaynet_1460000.pth

Or you can download from below link
https://drive.google.com/file/d/1M9L7yJJpDF5APDz4aJ_1w2TNdZJ4zcLB/view?usp=sharing, 
https://drive.google.com/file/d/1fLg1JKZLfDrunK3CzHwqAHlxHS_rqlek/view?usp=sharing

You can save the model anywhere as long as you adjust the path in the setting.json file

### Train

The project used the config file 'configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml'

Take note the name of the train and test dataset
```
DATASETS:
  TRAIN: ("doclaynet_train",)
  TEST: ("doclaynet_val",)

MAX_ITER: 1460000
IMS_PER_BATCH: 1
```
Go to 'tools/preprocess.py'
```
def register_dataset():
    register_coco_instances("doclaynet_val", {}, "/val_doclaynet.json", "/doclaynet/PNG/")
    register_coco_instances("doclaynet_train", {}, "/train_doclaynet.json", "/doclaynet/PNG/")
```
Change the path after downloaded the dataset

The MAX_ITER above will be the total literations in training and IMS_PER_BATCH is the number of images train in 1 literation. Please adjust base on your needs.

Below command to start training
```
python .\train.py --num-gpus 1 --config-file 'configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml'
```
To activate image augmentation, please use the trainWithImgAug() function in train.py

### Demo Application

The project using Django to build a simple website for demo purpose. 

Set the paths in the /web/predictionDir/setting.json
```
"PredictDocumentModel":/absolute/path/to/web/predictionDir/model/model_final_document_80000.pth,
"PredictLayoutModel":"/absolute/path/to/web/predictionDir/model/model_final_doclaynet_1460000.pth",
"TestImgDir":"/absolute/path/to/web/predictionDir/testImg/phototakinginput/",     -- path to test image file 
"TestDigitalImgDir":"/absolute/path/to/web/predictionDir/testImg/digitalinput/",      -- path to test image file 
"OutPutDocumentDir":/absolute/path/to/web/predictionDir/testImg/document/",
"OutPutLayoutDir":"/absolute/path/to/web/predictionDir/testImg/predict/",      -- path to output prediction for photo taking document image
"OutPutOriginDir":"/absolute/path/to/web/predictionDir/testImg/predictOrigin/",      -- path to output prediction for digital document image 
"ConfigFilePreidctOnDocument":"/absolute/path/to/web/predictionDir/model/configs/mask_rcnn_R_101_FPN_3x_1category.yaml",
"ConfigFilePreidctOnLayout":"/absolute/path/to/web/predictionDir/model/configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml",
"COCOJsonDocument":"/absolute/path/to/web/predictionDir/model/train_50_1c.json",
"COCOJsonLayout":"/absolute/path/to/web/predictionDir/model/val_doclaynet.json",
"PredDocumentThreshold":0.9,
"PredLayoutThreshold":0.5
```

```
cd /web/
python manage.py runserver
```
