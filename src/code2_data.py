import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
sys.path.append('../src')
import numpy as np
from numpy import asarray
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import streamlit as st

import json
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, model_from_json, load_model
from keras import backend as K

import h5py

model = h5py.File('models/model_v4red.h5', 'r')

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(face_cascade)
n = 0
counter_fotos = 0
def chorro(uploaded_file):
    counter_faces = 0
        
    img = Image.open(uploaded_file)
    new = img.save('demo/f"{counter_faces}".jpg')
    st.write("image saved")
    new_img = Image.open('demo/f"{counter_faces}".jpg')

    #input_img1 = cv2.imread(f"demo/{counter_faces}.jpg")
    input_img1 = cv2.imread('demo/f"{counter_faces}".jpg')
    input_img2 = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
    input_img3 = input_img2.copy()
    
    faceClass = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")
    faces = faceClass.detectMultiScale(input_img2,scaleFactor=1.1, minNeighbors=7)       
    for (x,y,w,h) in faces:
            
        st.write("got faces")
        counter_faces += 1
        img_data1 = input_img3 [y:y+h,x:x+w]
        img_data2 = cv2.resize (img_data1, (48, 48))
        img_data3 = np.stack(img_data2) 
        img_data4 = img_data2 / 255.0
        img_data5 = np.expand_dims(
            img_data4, axis=0).reshape(
                np.expand_dims(
                    img_data4, axis=0).shape[0], 48, 48, 1)
            
        cv2.imwrite(f"demo/{n}_{counter_faces}.jpg", img_data2)
            
        img_datashow = img_data3*255
        img_show = Image.fromarray(img_datashow)
        img = Image.open("images_support/cover1.jpeg") 
        img_show.save(f"demo/{n}_{counter_faces}_a.jpg")


        with open('models/model_v4red_weights.h5.json','r') as f:
            model_json = json.load(f)
            model = model_from_json(model_json)
            model.load_weights('models/model_v4red.h5')
            #model = h5py.File('models/model_v3.hdf5', 'r')
            EM = model.predict(img_data5)[0]
            model_red = load_model('model_v4red.h5')

            counter_faces = 0
            
            happy = EM[0]
            unhappy = EM[1]
            plt.imshow(Image.fromarray(EM.squeeze()*255))
            st.write("The prediction is… happy:{0:.5f} other:{1:.5f}".format(EM[0],EM[1]))
    return "TO BE CONTINUED"   
                
       
