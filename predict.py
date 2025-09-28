import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model
model = load_model('garbage_classifier.h5')

# Which image to test
img_path = 'metalgarbage.jpg'  # Change this to your image filename!

# Get class indices from your training generator
classes = sorted(os.listdir('garbage-dataset'))  # Make sure this matches your training folder!

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Rescale like in training
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

# Predict
pred = model.predict(img_array)
predicted_class_index = np.argmax(pred)
predicted_class = classes[predicted_class_index]

print(f"Prediction: {predicted_class} (confidence: {100*pred[0][predicted_class_index]:.2f}%)")
