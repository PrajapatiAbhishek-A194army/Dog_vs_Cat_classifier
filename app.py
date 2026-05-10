import streamlit as st
import tensorflow as tf
# from  tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import keras
model = keras.models.load_model("dog_cat_model.h5",compile=False)

st.title("Dog vs Cat Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)

    img = cv2.resize(img, (128, 128))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.success("Dog")
    else:
        st.success("Cat")