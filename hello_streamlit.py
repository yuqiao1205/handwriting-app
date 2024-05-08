import streamlit as st
from streamlit_drawable_canvas import st_canvas

import numpy as np
import cv2
np.set_printoptions(linewidth=1000)

#from mnistlib import infer
from mnistlib1_model import infer

# Load a pre-trained MNIST model


# Title
st.title("Team 2")
st.title("Handwriting Digit Recognition")

# Sidebar
st.sidebar.header("Draw a Digit")

# Create a canvas for drawing
canvas = st_canvas(height=280, width=280, stroke_width=30, background_color="#000000", stroke_color="#fefefe") # , stroke_color="#333333"

# Function to preprocess and predict the drawn digit
def predict_digit(image):
    # Resize the image to 28x28 pixels (MNIST input size)
    # Normalize pixel values to be between 0 and 1
    normalized_image = image / 255.0
    # Reshape the image to match the model's input shape
    input_data = np.reshape(normalized_image, (1, 28, 28))
    print(input_data.shape)

    # ******************Make a prediction using the loaded model********************
    predicted_digit = infer(input_data)
    return predicted_digit

# Button to make predictions
if st.sidebar.button("Recognize Digit"):
    # Get the drawn image from the canvas
    drawn_image = canvas.image_data[:, :, 0:3]

    # use cv2 to convert to grayscale
    drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(drawn_image, (28, 28))

    print(resized_image)

    # Predict the digit
    predicted_digit = predict_digit(resized_image)
    # Display the prediction result
    st.write(f"#### Predicted Digit: {predicted_digit}")


# Instructions
st.sidebar.markdown("1. Draw a digit in the canvas.")
st.sidebar.markdown("2. Click 'Recognize Digit' to see the prediction.")

# Info about the model
st.sidebar.markdown("Model: Pre-trained Convolutional Neural Network (CNN)")
st.sidebar.markdown("Dataset: MNIST")

# Example of an MNIST digit
# st.sidebar.image("mnist_example.png", caption="Example MNIST Digit")

# Clear button
if st.sidebar.button("Clear Canvas"):
    canvas.background_color = ""


