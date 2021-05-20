import numpy as np
import pandas as pd
import cv2
from PIL import Image

from keras.models import model_from_json
import json
import src.modelling as ccmo
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
n = 0
counter_fotos = 0

def transfImag2(path):
    print ('transforming image from {}'.format(path))

    input_img=cv2.imread(path)
    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(input_img, 1.25, 6)
    x,y,w,h = faces[0]
    img_data= input_img[y:y+h,x:x+w]
    img_data=cv2.resize(img_data,(48,48))
    
    img_data = np.stack(img_data)
    img_data = img_data / 255.0
    
    return img_data

def transf_imported_Image(foto_in):
    counter_fotos += 1
    counter_faces = 0
    img = Image.open(foto_in)
    img.save(f"demo/{counter_fotos}.jpg")

    input_img1 = cv2.imread(f"demo/{counter_fotos}.jpg")
    input_img2 = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
    input_img3 = input_img2.copy()

    faces = face_cascade.detectMultiScale(input_img2, 1.25, 6)
    
    for (x,y,w,h) in faces:
        counter_faces += 1
        img_data1 = input_img3 [y:y+h,x:x+w]
        img_data2 = cv2.resize (img_data1,(48,48))
        img_data3 = np.stack(img_data2) 
        img_data4 = img_data2 / 255.0
        img_data5 = img_data3 / 255.0
        cv2.imwrite(f"demo/{counter_fotos}_{counter_faces}.jpg",img_data2)
            
        img_data6 = np.expand_dims(img_data5,axis=0).reshape(np.expand_dims(img_data5,axis=0).shape[0], 48, 48, 1)
            
        img_datashow = img_data3*255
        img_show = Image.fromarray(img_datashow)
        #file_to_save = file.name.replace(".",f"_face{counter_faces}.")
        img_show.save(f"demo/{counter_fotos}_{counter_faces}_a.jpg")
            
        counter_faces = 0
            
        arr_for_model = img_data6
        print("terminado de proc")
        return arr_for_model
     






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