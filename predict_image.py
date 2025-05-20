import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('cnn_binary_classifier_model.h5')

# Set the path of the image you want to predict
img_path = r'C:\Users\vishw\OneDrive\Desktop\cats_and_dogs_filtered\sample.jpeg'


# Load and preprocess the image
img = image.load_img(img_path, target_size=(150, 150))  # resize to model's input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 150, 150, 3)
img_array = img_array / 255.0  # normalize like training

# Predict
prediction = model.predict(img_array)

# Since binary classification, output is probability between 0 and 1
if prediction[0][0] > 0.5:
    print("🔍 Prediction: Class 1 (%.2f%% confidence)" % (prediction[0][0]*100))
else:
    print("🔍 Prediction: Class 0 (%.2f%% confidence)" % ((1 - prediction[0][0])*100))
