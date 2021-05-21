from tensorflow.keras.models import model_from_json # Sequential, #load_model
import json
import os
import sys
sys.path.append('../src')
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import  load_model
from tensorflow.keras import backend as K
import h5py
from pathlib import Path

#import time
#import src.manage_data as ccst

#Prediction Function
def predic_(image1): 
    model = "../models/model_v3.hdf5"
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label 
"""
def predict(file):

    img_width, img_height = 48,48

    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model1.predict(x)
    result = array[0]
    #print(result)
    answer = np.argmax(result)
    return answer


if __name__ == "__predict__":
    predict()

def loadModel():
    with open("../models/model_v3.hdf5","r") as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    model.load_weights("../src/model_v3.hdf5")
    return model

"""
"""
def loadModel():
    with open('./'+const.MODEL_ROUTE+const.MODEL+'.h5.json','r') as f:
        model_json = json.load(f)

    model = model_from_json(model_json)
    model.load_weights('./'+const.MODEL_ROUTE+const.MODEL+'.h5')
    return model
"""
# importing tensorflow model libraries
