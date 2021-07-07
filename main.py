import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

import tensorflow as tf
import cv2

from src import main_app, sample_app, sandbox

PAGES = {
    "Sample page": sample_app, 
    "Main": main_app, 
    "Sandbox": sandbox, 
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio(
    "Go to", 
    list(PAGES.keys()), 
)

st.sidebar.write("確率が1.0になる場合、GradCamが機能しないバグあり...。")
st.sidebar.write("Mainから好きな画像をアップして楽しめます(犬顔か猫顔かの判定に使うもよし？)")

page = PAGES[selection]
page.app()



# # expander
# expander = st.beta_expander("問い合わせ")
# expander.write("問い合わせ内容を書く")

# # selectbox
# option = st.selectbox(
#     "あなたが好きな数字を教えてください", 
#     list(range(1, 11))
# )

# "あなたの好きな数字は、", option, "です。"

# # text
# option = st.text_input("あなたの趣味を教えてください。")
# "あなたの趣味は", option, "です。"

# # slider
# condition = st.slider("あなたの今の調子は？", 0, 100, 50)
# "コンディション", condition

# # file uploader
# uploaded_file = st.file_uploader("Choose an Image...", type="jpg")
# if uploaded_file is not None:
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded file", use_column_width=True)

# # progress bar
# def show_progress():
#     "Start!!"
#     latest_iteration = st.empty()
#     bar = st.progress(0)
#     for i in range(100):
#         latest_iteration.text(f"Iteration {i+1}")
#         bar.progress(i+1)
#         time.sleep(0.1)
#     "Done!!"
# show_progress()