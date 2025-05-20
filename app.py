import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import time
from io import BytesIO
import base64

# Load your trained model
model = load_model('cnn_binary_classifier_model.h5')

def preprocess(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stButton>button {
            background-color: #333 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Custom CSS styles for hover and animations
st.markdown("""
<style>
    /* Image zoom on hover */
    .zoom:hover {
        transform: scale(1.1);
        transition: transform 0.5s ease;
        cursor: zoom-in;
    }
    /* Button hover effect */
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
        transition: background-color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color:#4CAF50; text-align:center;'>🐶🐱 Cat vs Dog Classifier</h1>", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
- Upload a clear **cat** or **dog** image.
- Supported formats: jpg, jpeg, png.
- The app predicts if the image is a cat or dog.
- Use the **Exit** button below to close the app.
""")
st.sidebar.warning("⚠️ Upload only cat or dog images to get accurate results.")

# Show prediction history checkbox
show_history = st.sidebar.checkbox("🕒 Show Prediction History")

# Upload section
uploaded_file = st.file_uploader("Drag and drop or browse image", type=["jpg", "jpeg", "png"])

# Exit button
if st.button("🚪 Exit Application"):
    st.warning("Exiting application...")
    st.stop()

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Convert image to base64 for embedding
    img_base64 = image_to_base64(img)

    # Show uploaded image with zoom effect using HTML
    zoom_image_html = f"""
    <img src="data:image/png;base64,{img_base64}" class="zoom" style="max-width: 100%;"/>
    """
    st.markdown(zoom_image_html, unsafe_allow_html=True)

    # Show image details
    st.write(f"Image details: Format={img.format}, Size={img.size}, Mode={img.mode}")

    # Animated progress bar for prediction
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)  # adjust speed of animation
        progress_bar.progress(percent_complete + 1)

    # Predict
    img_array = preprocess(img)
    prediction = model.predict(img_array)[0][0]

    prob_dog = prediction
    prob_cat = 1 - prediction
    class_label = "Dog" if prob_dog > prob_cat else "Cat"
    confidence = max(prob_dog, prob_cat)

    st.write(f"🐱 Cat probability: {prob_cat*100:.2f}%")
    st.write(f"🐶 Dog probability: {prob_dog*100:.2f}%")

    if confidence < 0.6:
        st.warning("⚠️ Confidence is low, try uploading a clearer image.")

    if class_label == "Cat":
        st.success(f"🐱 Prediction: Cat with {confidence*100:.2f}% confidence!")
        st.info("😺 Did you know? Cats sleep for 12-16 hours a day!")
    else:
        st.info(f"🐶 Prediction: Dog with {confidence*100:.2f}% confidence!")
        st.info("🐕 Fun fact: Dogs have a sense of smell 40 times better than humans!")

    # Add to history
    st.session_state.history.append((uploaded_file.name, class_label, confidence))

    # Download prediction result as text
    result_text = f"Image: {uploaded_file.name}\nPrediction: {class_label}\nConfidence: {confidence*100:.2f}%"
    st.download_button("💾 Download Result", result_text, file_name="prediction_result.txt")

else:
    st.info("Please upload an image file to get prediction.")

# Show prediction history sidebar
if show_history:
    st.sidebar.write("### Prediction History (last 5)")
    for idx, (fname, label, conf) in enumerate(reversed(st.session_state.history[-5:])):
        st.sidebar.write(f"{idx+1}. {fname} - **{label}** ({conf*100:.2f}%)")

# Footer
st.markdown("""
<hr>
<p style="text-align:center;font-size:12px;color:gray;">
Created by P.Vishwateja | Project: Cat vs Dog Classifier
</p>
""", unsafe_allow_html=True)
