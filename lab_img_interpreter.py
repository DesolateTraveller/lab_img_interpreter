import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from skimage import measure
import openpyxl

@st.cache_data(ttl="2h")
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary_image

@st.cache_data(ttl="2h")
def calculate_features(contours):
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            sphericity = 4 * np.pi * (area / (perimeter ** 2))
        else:
            sphericity = 0
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        features.append({
            "Aspect Ratio": aspect_ratio,
            "Sphericity": sphericity,
            "Area": area,
            "Perimeter": perimeter
        })
    return features

def create_dataframe(image_features):
    all_data = []
    for i, features in enumerate(image_features):
        for feature in features:
            all_data.append({
                "Image Number": i + 1,
                **feature
            })
    return pd.DataFrame(all_data)

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Image Features")
    processed_data = output.getvalue()
    return processed_data

# Main Streamlit App
st.title("Image Feature Extraction for Multiple Images")

# Allow users to upload multiple image files
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    image_features = []
    
    for uploaded_file in uploaded_files:
        # Load the image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Process the image and extract contours
        contours, binary_image = process_image(image)
        
        # Calculate features for the image
        features = calculate_features(contours)
        image_features.append(features)
    
    # Create a DataFrame to store all the features
    df = create_dataframe(image_features)
    
    # Display the DataFrame
    st.dataframe(df, use_container_width=True)

    # Provide option to download the data as an Excel file
    excel_data = convert_df_to_excel(df)
    st.download_button(
        label="📥 Download Data as Excel",
        data=excel_data,
        file_name="image_features.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
