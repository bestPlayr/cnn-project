import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle

model = load_model('cats_dogs.h5')

with open('label_encoder.pkl', 'rb') as encoder:
    label_encoder = pickle.load(encoder)

def predict_image(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    class_label = label_encoder.inverse_transform(predicted_class)
    return class_label[0]

st.title("Cats and Dogs Classifier")
st.write("This project is built using a Convolutional Neural Network (CNN) to classify images of cats and dogs.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    if st.button("Predict"):
        prediction = predict_image(image)
        st.write(f"Predicted Class: **{prediction}**")
