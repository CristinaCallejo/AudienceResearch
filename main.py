import streamlit as st

import numpy as np
import pandas as pd
import json
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
import src.code2_data as cc2
#import src.code3_data as cc3

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.metrics import categorical_accuracy
import tensorflow.keras.backend as K
import streamlit.components.v1 as components
import h5py




model = h5py.File('models/model_v3.hdf5', 'r')

# LANDING
img = Image.open("images_support/cover1.jpeg")
st.image(img)
st.title("Focusing on our audience")
st.header("Customary quote")
st.markdown("> I just love to go home, no matter where I am [...]")

# HOOK
st.write(
    """What did your audience think about the show?
    Analyzing reactions with captured images of your clients during the show.
    """)

# INPUT IMAGE FROM USER
uploaded_file = st.file_uploader(
    "each image goes in here...", 
    type = ['jpeg', 'jpg', 'png']
    )

n = 0

if uploaded_file:
    n +=1
    foto = Image.open(uploaded_file)
    fot = foto.save(f"demo/foto{n}.jpg")
    fot_pth = Path.cwd()/f"demo/foto{n}.jpg"
    st.image(foto, caption="Here it is", use_column_width=True)
    st.write("")
    st.write("Processing...")

# IS THE USER/VIEWER HAPPY? 
   
    # ...formatting input 
    cc2.chorro(uploaded_file)
        