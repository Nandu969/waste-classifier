import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import pandas as pd

# --- Region-specific disposal instructions sample ---
disposal_rules = {
    "Hyderabad": {
        "plastic":      "Recycle at designated points.",
        "battery":      "Hazardous—take to e-waste center.",
        "metal":        "Recycle in dry waste bin.",
        "glass":        "Recycle in dry waste bin.",
        "cardboard":    "Recycle or compost if clean.",
        "trash":        "Landfill only.",
        "clothes":      "Donate or textile recycling center.",
        "shoes":        "Donate or discard in landfill.",
        "paper":        "Recycle if clean; landfill if dirty.",
        "biological":   "Compost at home or use wet waste bin."
    },
    "Chennai": {
        "plastic":      "Landfill; limited recycling.",
        "battery":      "E-waste center only.",
        "metal":        "Recycle if facility available.",
        "glass":        "Recycle at glass collection points.",
        "cardboard":    "Recycle or landfill if dirty.",
        "trash":        "Landfill.",
        "clothes":      "Donate or take to textile recycling.",
        "shoes":        "Landfill after cleaning.",
        "paper":        "Recycle if clean.",
        "biological":   "Put in green waste bin."
    }
    # Add more regions and rules as needed.
}

regions = list(disposal_rules.keys())

# Load model and class names
model = load_model('garbage_classifier.h5')
classes = sorted([d for d in os.listdir('garbage-dataset') if os.path.isdir(os.path.join('garbage-dataset', d))])

st.title("Garbage Image Classifier")
st.write("Upload an image to get a prediction and disposal instructions specific to your region!")

# Choose region for guidance
region = st.selectbox("Select your city/region for disposal rules:", regions)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    top_idx = np.argmax(pred)
    predicted_class = classes[top_idx]

    st.success(f"Prediction: **{predicted_class}** ({100*pred[top_idx]:.2f}%)")
    # Region-specific disposal guidance
    disposal = disposal_rules.get(region, {}).get(predicted_class, "No info available for this region/class.")
    st.info(f"**Disposal advice for {region}:** {disposal}")

    # --- User Feedback section ---
    st.write("#### Was this prediction correct?")
    feedback = st.radio("Give feedback:", ("Yes", "No"), key="feedback_radio")
    if feedback == "No":
        correct_class = st.selectbox("What is the correct class?", classes, key="correct_class")
    else:
        correct_class = predicted_class

    # Save feedback when button is pressed
    if st.button("Submit Feedback"):
        with open("feedback_log.csv", "a") as f:
            f.write(f"{uploaded_file.name},{predicted_class},{correct_class},{region},{feedback}\n")
        st.success("Thank you! Your feedback has been recorded.")

st.sidebar.markdown("### About")
st.sidebar.write("""
**Garbage Classifier Web App**

- Model: MobileNetV2 Transfer Learning
- Region-aware disposal guidance
- User feedback for continual improvement
- Upload an image and select your city for custom advice!
""")

# --- Feedback Viewing/Download Section ---
st.sidebar.write("---")
st.sidebar.write("### View Collected Feedback")
if os.path.exists("feedback_log.csv"):
    if st.sidebar.button("Show Feedback"):
        df = pd.read_csv(
            "feedback_log.csv",
            names=["Image", "Predicted Class", "User Class", "Region", "Correct?"]
        )
        st.sidebar.dataframe(df)
        st.sidebar.download_button(
            label="Download Feedback CSV",
            data=df.to_csv(index=False),
            file_name="feedback_log.csv",
            mime="text/csv"
        )
else:
    st.sidebar.info("No feedback logged yet.")
