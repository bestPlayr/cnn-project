import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
model = load_model('cats_dogs.h5')


with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def predict_image(image):
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (128, 128))  
    img = img / 255.0  
    img = np.array(img)  
    img = np.expand_dims(img, axis=0) 

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)  
    class_label = label_encoder.inverse_transform(predicted_class) 
    return class_label[0]  


st.title("Cat and Dog Classifier")
st.write("This project is built using a  Convolutional Neural Network (CNN) to classify images of cats and dogs.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  
    st.image(image, channels="BGR", caption="Uploaded Image")
    if st.button("Predict"):
        prediction = predict_image(image)
        st.write(f"Predicted Class: **{prediction}**")