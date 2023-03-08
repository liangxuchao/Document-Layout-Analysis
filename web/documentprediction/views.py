import base64
from io import BytesIO
import json
import pathlib
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.template import loader
import requests
import os
import csv
from pathlib import Path
import pandas as pd

import os
from django.core.files.storage import FileSystemStorage
from predictionDir.advancePrediction import documentPreditor
arr = os.listdir()
dir = os.path.dirname(__file__)
from PIL import Image
predictionSettingFile = "C:/Users/Liang Xu Chao/Documents/ntu/project/web/predictionDir/setting.json"
with open(predictionSettingFile) as user_file:
        file_contents = user_file.read()
    
parsed_json = json.loads(file_contents)
def index(request):
    
    imglist = [f for f in os.listdir(parsed_json["TestImgDir"]) if os.path.isfile(os.path.join(parsed_json["TestImgDir"], f))]
    imgArr = []
    for x in range(0,len(imglist)):
        item =[]

        image = Image.open(os.path.join(parsed_json["TestImgDir"],imglist[x]))
        image64 = image_to_base64(image)
        # arr.append(image64)
        item.append(imglist[x])
        item.append(image64)
        imgArr.append(item)
    
    return render(request, 'index.html',{'config':parsed_json,'predimglist':imgArr})

def predictdigital(request):
    imglist = [f for f in os.listdir(parsed_json["TestDigitalImgDir"]) if os.path.isfile(os.path.join(parsed_json["TestDigitalImgDir"], f))]
    imgArr = []
    for x in range(0,len(imglist)):
        item =[]

        image = Image.open(os.path.join(parsed_json["TestDigitalImgDir"],imglist[x]))
        image64 = image_to_base64(image)
        # arr.append(image64)
        item.append(imglist[x])
        item.append(image64)
        imgArr.append(item)
    return render(request, 'predictdigital.html',{'config':parsed_json,'predimglist':imgArr})

def uploadtestimg(request):
    # request should be ajax and method should be POST.
    if request.method == "POST":
        # get the form data
        upload_file = request.FILES.getlist('files[]', None)
        print(upload_file)
        for f in upload_file:
            file_extension = pathlib.Path(f.name).suffix
            print(file_extension)
            # save the data and after fetch the object in instance
            if file_extension.lower() in ['.png','.jpg','.jpeg']:
                 
                    fs = FileSystemStorage()
                    if request.POST.get('type') == 'digital':
                         
                        fs.save(os.path.join(parsed_json["TestDigitalImgDir"], f.name), f)
                    else:
                    
                        fs.save(os.path.join(parsed_json["TestImgDir"], f.name), f)
                    
                    # send to client side.
                    return JsonResponse({"success": "Upload success"}, status=200)
        
            else:
                # some form errors occured.
                return JsonResponse({"error": "Please upload an image!"}, status=400)

    # some error occured
    return JsonResponse({"error": "Unexpected error occur!"}, status=400)

def apredict(request):
    
    returnlayoutimgs =[]
    if request.method == "POST":
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
        print(request.POST.get('type'))
        if request.POST.get('type') == 'digital':
            
            ourtputdir = parsed_json["OutPutOriginDir"]
            predictor.simplePredict(parsed_json["TestDigitalImgDir"])
            
        else:
           
            ourtputdir = parsed_json["OutPutLayoutDir"]
            predictor.predictImages(parsed_json["TestImgDir"])
            
        predlayout  = [f for f in os.listdir(ourtputdir) if os.path.isfile(os.path.join(ourtputdir, f))]
            
        print(predlayout)
        for x in range(0,len(predlayout)):
            # arr = []
            # image = Image.open(os.path.join(parsed_json["TestImgDir"],imgs[x]))
            # image64 = image_to_base64(image)
            # arr.append(image64)

            image = Image.open(os.path.join(ourtputdir,predlayout[x]))
            image64 = image_to_base64(image)
            # arr.append(image64)
            returnlayoutimgs.append(image64)
        
    return JsonResponse({"success": "Finish Prediction","returnlayoutimgs":returnlayoutimgs}, status=200)
def image_to_base64(image):
    buff = BytesIO()
    image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    img_str = img_str.decode("utf-8")  # convert to str and cut b'' chars
    return img_str         