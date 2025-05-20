# Cat-vs-Dog-Classifier
# 🐶🐱 Cat vs Dog Classifier using CNN

A simple and interactive **Streamlit** web app that allows users to upload an image of a **cat** or **dog** and classifies it using a trained **Convolutional Neural Network (CNN)** model built with **TensorFlow/Keras**

Try it locally by running:
bash
streamlit run app.py

# project structure (you can arrange the structure as you like,with the required files and folders but this is the basic fine structure but "i used my own")
├── cnn_binary_classifier_model.h5     # Trained CNN model
├── app.py                             # Main Streamlit app
├── README.md                          # This file
├── requirements.txt                   # Python dependencies

🔧 Features
✅ Upload image of a cat or dog
✅ Image zoom effect on hover
✅ Animated progress bar for prediction
✅ Confidence level display
✅ Prediction with visual feedback
✅ Sidebar with help, sample images, and exit button
✅ Responsive design using HTML/CSS with Streamlit

🧠 Model Info
-Trained using TensorFlow/Keras
-Binary classification: Cat (0) vs Dog (1)
-Input image size: 150x150 pixels
-Model file: cnn_binary_classifier_model.h5

🚀 How This Project Was Built – Step-by-Step Guide
This section explains the full journey of how the Cat vs Dog Classifier was developed, trained, and deployed using Streamlit.

🔸 Step 1: Dataset Collection
The dataset used is typically the Dogs vs Cats dataset provided by Kaggle.
It contains two folders:
/train: with images labeled as either cat or dog.
/test: for final evaluation (optional).
📦 Example dataset: https://www.kaggle.com/c/dogs-vs-cats/data


🔸 Step 2: Preprocessing the Dataset
Load the dataset using libraries like os, PIL, NumPy, and TensorFlow.
Resize images to a fixed size, such as 150x150 pixels.
Normalize pixel values (divide by 255).
Split the data into training and validation sets (e.g., 80/20 split).
Use ImageDataGenerator from Keras for:
Rescaling
Augmentation (optional)


🔸 Step 3: Building the CNN Model

🔸 Step 4: Training the Model
Train the CNN using the training and validation sets.

🔸 Step 5: Creating the Streamlit Web App
Install Streamlit
Create a file named app.py.
Add logic to:
Load the trained model
Upload and display image
Preprocess the image
Predict and show results with progress bar and styling
Add CSS for button hover and image zoom


🔸 Step 7: Testing the App
Upload various cat and dog images.
Observe predictions and confidence levels.
If needed, retrain the model for better accuracy.

then its done your project is done....





