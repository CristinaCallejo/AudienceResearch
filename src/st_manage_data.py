import os
import sys
sys.path.append('../src')
import numpy as np
from numpy import asarray
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

import json
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, model_from_json
from keras import backend as K
#import src.preds as mod


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

n = 0
counter_fotos = 0

def predic_(path):

    input_img=cv2.imread(str(path))
    input_img2=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(input_img2, 1.25, 6)
    
    x,y,w,h = faces[0]
    img_data1= input_img2[y:y+h,x:x+w]
    img_data2= cv2.resize(img_data1,(48,48))
    
    img_data3 = np.stack(img_data2)
    img_data4 = img_data3 / 255.0
    
    return img_data4
"""
    def transf_imported_Image(uploaded_file):
        counter_faces = 0
        img = Image.open(uploaded_file)
        new = img.save(f"demo/{counter_faces}.jpg")

        #input_img1 = cv2.imread(f"demo/{counter_faces}.jpg")
        input_img1 = cv2.imread(new)
        input_img2 = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
        input_img3 = input_img2.copy()
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(input_img2, 1.25, 6)
        
        for (x,y,w,h) in faces:
            counter_faces += 1
            img_data1 = input_img3 [y:y+h,x:x+w]
            img_data2 = cv2.resize (img_data1,(48,48))
            img_data3 = np.stack(img_data2) 
            img_data4 = img_data2 / 255.0
            cv2.imwrite(f"demo/{counter_fotos}_{counter_faces}.jpg",img_data2)
            return img_data4
"""


"""    
        img_data6 = np.expand_dims(img_data5,axis=0).reshape(np.expand_dims(img_data5,axis=0).shape[0], 48, 48, 1)
            
        img_datashow = img_data3*255
        img_show = Image.fromarray(img_datashow)
        #file_to_save = file.name.replace(".",f"_face{counter_faces}.")
        img_show.save(f"demo/{counter_fotos}_{counter_faces}_a.jpg")
            
        counter_faces = 0
            
        arr_for_model = img_data6
        print("terminado de proc")
        return arr_for_model
"""
     






def carga_data():
    data = pd.read_csv("data/clean.csv")
    return data


def grafico_barras_st():
    data = carga_data()
    data = data.groupby("character_name").agg({"character_name":'count'}).rename(columns={"character_name":"character_name", "character_name":"número de frases"}).reset_index().set_index("character_name", drop=True)
    return data


def lista_personajes():
    data = carga_data()
    return list(data.character_name.unique())


def grafico(personaje):
    data = carga_data()
    data = data[(data["character_name"]== f"{personaje}")]
    return data

def bar_2():
    data = carga_data()
    data = data.groupby("character_name").agg({'polarity': 'mean'}).reset_index().set_index("character_name", drop=True)
    return data

"""
conv to arr

"""