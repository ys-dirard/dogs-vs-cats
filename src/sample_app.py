import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from io import StringIO

import tensorflow as tf
# import keras
import cv2

from src.grad_cam import grad_cam

def app():
    # magic command
    
    st.markdown("# Dogs vs Cats")
    

    # # display image
    # st.write("Display Image")
    # img = Image.open("./1.jpg")
    # # checkbox
    # if st.checkbox("Show Image"):
    #     st.image(img, caption="dog", use_column_width=True)


    # side bar
    # st.sidebar...
    model_path = "./models/best_functional_model.hdf5"
    model = tf.keras.models.load_model(model_path)

    cols=["name", "(type)", "output shape"]
    layers = model.layers
    layer_names = []
    layer_types = []
    layer_output_shapes = []
    for l in layers:
        layer_names.append(l.name)
        layer_types.append(l.__class__.__name__)
        layer_output_shapes.append(l.output_shape)
    df_summary = pd.DataFrame({
        cols[0]: layer_names, 
        cols[1]: layer_types, 
        cols[2]: layer_output_shapes
        })

    # expander
    expander = st.beta_expander("Model Architecture (Summary)")
    expander.table(df_summary)

    # uploaded_file = st.file_uploader("Please upload dog or cat picture (.jpg).", type="jpg")

    img = Image.open("./data/dog.1.jpg")
    # uploaded_file = st.image(img, caption="dog", use_column_width=True)

    # 2columns
    left_column, right_column = st.beta_columns(2)

    if True:
        # img = Image.open(uploaded_file)
        img = cv2.resize(np.array(img), (150, 150))
        left_column.image(img, caption="Uploaded file (resized)", use_column_width=True)

        pred_result = model.predict(np.array([img / 255.]))
        ans = ""
        score = 0
        if pred_result[0][0]>0.5:
            ans = "dog"
            score = pred_result[0][0]
        else:
            ans="cat"
            score = 1-pred_result[0][0]

        message = f"label: {ans} <br> probability: {score}"
        right_column.write("Prediction result")
        right_column.write(f"label: {ans}")
        right_column.write(f"probability: {score}")


        st.markdown("## GradCam Result")
        # 5columns
        conv_layers_list = [
            ["block5_conv3", "block5_conv2", "block5_conv1"], 
            ["block4_conv3", "block4_conv2", "block4_conv1"], 
            ["block3_conv3", "block3_conv2", "block3_conv1"], 
            ["block2_conv2", "block2_conv1"], 
            ["block1_conv2", "block1_conv1"]
            ]
        columns = st.beta_columns(len(conv_layers_list[0]))
        for i, layers in enumerate(conv_layers_list):
            for j, layer in enumerate(layers):
                cam = grad_cam(model, img, layer)
                dst = tf.keras.preprocessing.image.array_to_img(cam)
                columns[j].image(dst, caption=layer, use_column_width=True)
