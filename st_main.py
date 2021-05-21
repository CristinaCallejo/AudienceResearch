import streamlit as st

import numpy as np
import pandas as pd

import os
import sys
sys.path.append('../src')
from pathlib import Path
from PIL import Image
import cv2
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import src.st_manage_data as ccc
import src.preds as mod
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#import codecs
import streamlit.components.v1 as components


# LANDING
img = Image.open("images_support/cover1.jpeg")
st.image(img)

st.title("Focusing on our audience")
st.header("Customary quote")
st.markdown("> I just love to go home, no matter where I am [...]")
st.write(
    """What did your audience think about the show?
    Analyzing reactions with captured images of your clients during the show.
    """)

uploaded_file = st.file_uploader("Drag & Drop your demo image here...", type = ['jpeg', 'jpg', 'png'])
n = 0
if uploaded_file:
    n +=1
    foto = Image.open(uploaded_file)
    foto.save(f"demo/foto{n}.jpg")
    foto_pth = Path.cwd()/f"demo/foto{n}.jpg"
    st.image(foto, caption="Let's find out", use_column_width=True)
    st.write("")
    st.write("Processing...")

    happy_or_not = ccc.predic_(foto_pth)

    st.write('%s (%.2f%%)' % (label[1], label[2]*100))
"""
if uploaded_file:
    n+=1
    PIC = ccst.transf_imported_Image(uploaded_file)
    input_img=cv2.imread(uploaded_file) # get the array of the original pic

    plt.subplot(121)
    plt.imshow(input_img) # original pic

    plt.subplot(122)
    plt.imshow(Image.fromarray(PIC.squeeze()*255)) # transformed pic

"""    
"""
PIC = transfImag2('foto.jpg') # transform pic
input_img=cv2.imread('foto.jpg') # get the array of the original pic

plt.subplot(121)
plt.imshow(input_img) # original pic
print(input_img.shape)
plt.subplot(122)
plt.imshow(Image.fromarray(PIC.squeeze()*255)) # transformed pic

PIC = np.expand_dims(PIC,axis=0).reshape(np.expand_dims(PIC,axis=0).shape[0], 48, 48, 1)
print(PIC.shape)
pred2 = model_1.predict(PIC)[0]
print("Probs -> happy:{0:.5f} unhappy:{1:.5f}".format(pred2[0],pred2[1]))

happy = pred2[0]
unhappy = pred2[1]



st.write("""
# Mi súper Dashboard
#Con Jake el perro y Finn el humano lo pasaremos guaaaaay
""")

st.dataframe(dat.carga_data())

st.write("""
#Gragiquito de barras propio de streamlit
""")


st.dataframe(dat.grafico_barras_st())
datos = dat.grafico_barras_st()
st.bar_chart(datos)

st.write("""
#Gráfico de Plotly
""")

personaje = st.selectbox(

    "Selecciona un personaje", dat.lista_personajes()
)

datagraf = dat.grafico(personaje)

fig = px.line(datagraf, y="polarity", title = f"Evolución de la polaridad de {personaje}")

st.plotly_chart(fig)

st.write("""#Formulario de texto""")
texto = st.text_input("Lo que tiene que introducir", "Texto por defecto")
st.write("Ha introducido ", texto)


st.write ("""Gestor de archivos""")

uploaded_file = st.file_uploader("Sube un csv")

if uploaded_file: 
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

foto = st.file_uploader("Sube una foto")
if foto:
    imagen = Image.open(foto)
    imagen.save("data/foto.png")
    st.write ("tu foto se ha subido correctamente")


st.write("Columnas")

#datines = dat.bar_2()

col1,col2 = st.beta_columns([4,2])

col1.subheader("El Gráfico")
#col1.bar_chart(datines)

col2.subheader("Los datos")
#col2.write(datines)

"""

map_1 = folium.Map(location = [45.50935, -73.57225], zoom_start = 15)
folium_static(map_1)


archivo = codecs.open("data/mapa.html", "r")
mapa = archivo.read()
components.html(mapa, height = 550)
"""