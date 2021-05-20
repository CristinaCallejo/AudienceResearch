from tensorflow.keras.models import model_from_json,load_model
import json
import src.manage_data as ccst
import numpy as np

def loadModel():
    with open("../src/model_v2_epoch30happyunhappy.hdf5"),'r') as f:
    model_json = json.load(f)
model = model_from_json(model_json)
model.load_weights("../src/model_v2_epoch30happyunhappy.hdf5")

def loadModel():
    with open('./'+const.MODEL_ROUTE+const.MODEL+'.h5.json','r') as f:
        model_json = json.load(f)

    model = model_from_json(model_json)
    model.load_weights('./'+const.MODEL_ROUTE+const.MODEL+'.h5')
    return model
# importing tensorflow model libraries

import json
import time