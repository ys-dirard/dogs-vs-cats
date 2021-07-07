import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time

def app():
    st.title("Streanimlit introduction")

    st.write("DataFrame")

    df = pd.DataFrame({
        "1 col": [1, 2, 3, 4], 
        "2 col": [10, 20, 30, 40], 
    })

    st.write(df)

    # create dynamic table
    # st.dataframe(df.style.highlight_max(axis=0))
    # width=100, height=100も指定可能

    # create static table
    st.table(df)

    # →API reference displaydata

    # magic command
    """
    # chapter
    ## section
    ### subsection
    $\sin x$

    ```python
    import streamlit as st
    ```
    """

    # chart
    df = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["a", "b", "c"]
    )

    st.line_chart(df)

    st.area_chart(df)

    st.bar_chart(df)

    # map
    df = pd.DataFrame(
        np.random.randn(100, 2) /[50, 50] + [35.69, 139.70],
        columns=["lat", "lon"]
    )

    st.map(df)

    # display image
    st.write("Display Image")
    img = Image.open("./data/dog.1.jpg")
    # checkbox
    if st.checkbox("Show Image"):
        st.image(img, caption="dog", use_column_width=True)


    # side bar
    # st.sidebar...

    # 2columns
    left_column, right_column = st.beta_columns(2)

    button = left_column.button("右カラムに文字を表示")
    if button:
        right_column.write("ここは右カラム")

    # expander
    expander = st.beta_expander("問い合わせ")
    expander.write("問い合わせ内容を書く")

    # selectbox
    option = st.selectbox(
        "あなたが好きな数字を教えてください", 
        list(range(1, 11))
    )

    st.write("あなたの好きな数字は、", option, "です。")

    # text
    option = st.text_input("あなたの趣味を教えてください。")
    st.write("あなたの趣味は", option, "です。")

    # slider
    condition = st.slider("あなたの今の調子は？", 0, 100, 50)
    st.write("コンディション", condition)

    # file uploader
    uploaded_file = st.file_uploader("Choose an Image...", type="jpg")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded file", use_column_width=True)

    # progress bar
    def show_progress():
        st.write("Start!!")
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f"Iteration {i+1}")
            bar.progress(i+1)
            time.sleep(0.1)
        st.write("Done!!")
    show_progress()