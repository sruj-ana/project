#!/usr/bin/env python3
from fastapi import FastAPI, Response, Request, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras import models
import numpy as np
from PIL import Image
import os
import shutil
import time
import tensorflow_addons as tfa


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app = FastAPI()

# Instanciate cache object to mem store models
cache_models = {}


def read_imagefile_classif(file):
    img_classif = Image.open(file)
    img_classif = img_classif.resize((32,32),Image.ANTIALIAS)
    img_classif = np.array([np.array(img_classif)])
    return img_classif


def read_imagefile(file):
    img = Image.open(file)
    img = img.resize((256,256),Image.ANTIALIAS)
    img = np.array([np.array(img)])
    img = img / 255
    return img

# ---------------------------------------------------------------------------
# - Handle a POST request to handle an image -
# ---------------------------------------------------------------------------
def check_extension(filename):
    ALLOWED_EXTENSION = ["jpg", "jpeg", "png"]
    # Extract extension
    extension = filename.split(".")[-1:][0].lower()
    if extension not in ALLOWED_EXTENSION :
        return False
    else :
        return True


class PredictPayload(BaseModel):
    key : str
    value : int

@app.on_event("startup")
async def startup_event():
    '''
    On api startup, load and store models in mem
    '''
    print("load model ...")
    dirname = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dirname,'models','model_v5')
    model_0 = models.load_model('model0_classif_eye.h5')
    model_1 = models.load_model('model1_models_model_inception2_v0_N_70.h5')
    model_2 = models.load_model('model2_models_model_vlundi_v1_GCAHMO_58.h5')
    cache_models["model_0"] = model_0
    cache_models["model_1"] = model_1
    cache_models["model_2"] = model_2
    print("models are ready ...")


@app.post("/predict")
async def predict_handler(response : Response, inputImage : UploadFile = File(...)):
    '''
    Check extension
    '''
    check = check_extension(inputImage.filename)

    if check == False :
        response_payload = {
                "status" : "error",
                "message" : "Input file format not valid"
                }
        response.status_code=400
        return response

    '''
    Temp image
    '''
    temp_image_classif = str(int(time.time())) + "_" + "classif" + "_" + inputImage.filename
    #shutil.copyfile(inputImage.filename, temp_image_classif)

    #print(temp_image_classif)
    with open(temp_image_classif, "wb") as buffer:
        shutil.copyfileobj(inputImage.file, buffer)

    inputImage.close()
    buffer.close()


    temp_image = str(int(time.time())) + "_" + inputImage.filename
    shutil.copyfile(temp_image_classif, temp_image)
    # with open(temp_image, "wb") as buffer1:
    #     shutil.copyfileobj(inputImage.file, buffer1)
    # inputImage.close()
    # buffer1.close()

    '''
    Prediction worker
    '''

    img_0 = read_imagefile_classif(temp_image_classif)
    model_0 = cache_models["model_0"]
    pred_0 = model_0.predict(img_0)

    if int(pred_0[0][0]) == 1:
        #temp_image = "/tmp/" + str(int(time.time())) + "_" + inputImage.filename
        # temp_image = str(int(time.time())) + "_" + inputImage.filename
        # #shutil.copyfile(inputImage.filename, temp_image)
        # with open(temp_image, "wb") as buffer1:
        #     shutil.copyfileobj(inputImage.file, buffer1)
        # inputImage.close()
        # buffer1.close()


        # Extraction image
        #img_classif = read_imagefile_classif(temp_image_classif)
        img_1 = read_imagefile(temp_image)

        # load cached model
        model_1 = cache_models["model_1"]

        # prediction
        pred_1 = model_1.predict(img_1)
        pred_1_arg = np.argmax(pred_1)

        if pred_1_arg == 1:
            # Extraction image
            # img_2 = read_imagefile(temp_image)
            img_2 = img_1
            # load cached model
            model_2 = cache_models["model_2"]

            # prediction
            pred_2 = model_2.predict(img_2)

            response_payload = {
                    'prediction' : pred_2[0].tolist()
                    }
            response.status_code = 202
            response.headers["Content-Type"] = "application/json"

            if os.path.exists(temp_image_classif):
                os.remove(temp_image_classif)

            if os.path.exists(temp_image):
                os.remove(temp_image)

            return response_payload


        else:
            response_payload = {
                "message_normal" : "Normal"
                }
            response.status_code = 201
            response.headers["Content-Type"] = "application/json"

            if os.path.exists(temp_image_classif):
                os.remove(temp_image_classif)

            if os.path.exists(temp_image):
                os.remove(temp_image)

            return response_payload['message_normal']

    else:
        response_payload = {
                "message" : "Not an eye, upload again !"
                }
        response.status_code=200

        if os.path.exists(temp_image_classif):
                os.remove(temp_image_classif)

        if os.path.exists(temp_image):
                os.remove(temp_image)

        return response_payload['message']

    '''
    Delete temp image
    '''
    if os.path.exists(temp_image_classif):
        os.remove(temp_image_classif)

    if os.path.exists(temp_image):
        os.remove(temp_image)
