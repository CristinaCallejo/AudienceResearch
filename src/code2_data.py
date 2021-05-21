import os
import sys
from matplotlib.pyplot import imshow
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
sys.path.append('../src')
import numpy as np
from numpy import asarray
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

import json
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, model_from_json, load_model
from keras import backend as K
#import src.preds as mod
import h5py

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

n = 0
counter_fotos = 0
"""
def predict_(path):

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

    # ...formatting input 
    binary_emo = cc2.transf_imported_Image(uploaded_file)
    
    # ...let's see what science can tell us!
    being_happy = cc2.playingGod(model.predict(binary_emo)[0])

    try: 	model_path = './models/model.h5'
	        model_weights_path = './models/weights.h5'
    
    happiness = being_happy[0]
    not_quite_so_much_happiness = being_happy[1]
    
    if being_happy[0] > 0.7:


"""

""

def playingGod(happiness):
    model = load_model()
    
    with open()
    model.load_weights(weights_path)
    Y_pred = model.predict(X_test)
    # Convert predictions classes to one hot vectors 
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    # Convert validation observations to one hot vectors
    print(Y_pred_classes)
    Y_true = np.argmax(y_test,axis = 1)
    print(Y_true)
    # compute the confusion matrix
    cm = confusion_matrix(Y_true, Y_pred_classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot the confusion matrix
    f,ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, linewidths=0.01,cmap="YlGnBu",linecolor="gray", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    


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
        img_data2 = cv2.resize (img_data1, (48, 48))
        img_data3 = np.stack(img_data2) 
        img_data4 = img_data2 / 255.0
        img_data5 = np.expand_dims(
            img_data4, axis=0).reshape(
                np.expand_dims(
                    img_data4, axis=0).shape[0], 48, 48, 1)
        
        cv2.imwrite(f"demo/{counter_fotos}_{counter_faces}.jpg", img_data2)
        
        img_datashow = img_data3*255
        img_show = Image.fromarray(img_datashow)
        img = Image.open("images_support/cover1.jpeg")
        img_show.save(f"demo/{counter_fotos}_{counter_faces}_a.jpg")
            
        arr_for_model = img_data5
        counter_faces = 0
        
        print("terminado de proc")
        
        return arr_for_model

















        return img_data5
        img_datashow = img_data3*255
        img_show = Image.fromarray(img_datashow)
        #file_to_save = file.name.replace(".",f"_face{counter_faces}.")
        img_show.save(f"demo/{counter_fotos}_{counter_faces}_a.jpg")
            
        counter_faces = 0
            
        arr_for_model = img_data5
        print("terminado de proc")
        return arr_for_model
"""

            def load_trained_model(weights_path):
        model = create_model()
   model.load_weights(weights_path)
   
     






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


conv to arr

"""