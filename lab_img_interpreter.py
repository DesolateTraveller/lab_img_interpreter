import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import openpyxl

# Function to process each image and extract features
def process_image(img_array):
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)                                # Convert the image to grayscale
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)                                       # Apply Gaussian blur to reduce noise

    # OTSU Thresholding
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    otsu_valid_particles = 0
    otsu_sphericities = []
    otsu_aspect_ratios = []
    for contour in otsu_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect_ratio = min(width, height) / max(width, height)
            otsu_aspect_ratios.append(aspect_ratio)
        if perimeter > 0:
            sphericity = 4 * np.pi * (area / (perimeter ** 2))
            otsu_sphericities.append(sphericity)
            otsu_valid_particles += 1
    
    otsu_avg_sphericity = np.mean(otsu_sphericities) if otsu_sphericities else 0
    otsu_avg_aspect_ratio = np.mean(otsu_aspect_ratios) if otsu_aspect_ratios else 0

    # Canny Edge Detection
    canny_edges = cv2.Canny(blurred, 80, 170)
    canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    canny_valid_particles = 0
    canny_sphericities = []
    canny_aspect_ratios = []
    for contour in canny_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect_ratio = min(width, height) / max(width, height)
            canny_aspect_ratios.append(aspect_ratio)
        if perimeter > 0:
            sphericity = 4 * np.pi * (area / (perimeter ** 2))
            canny_sphericities.append(sphericity)
            canny_valid_particles += 1
    
    canny_avg_sphericity = np.mean(canny_sphericities) if canny_sphericities else 0
    canny_avg_aspect_ratio = np.mean(canny_aspect_ratios) if canny_aspect_ratios else 0
    
    return {
        "OTSU": {
            "Valid Particles": otsu_valid_particles,
            "Total Particles": len(otsu_contours),
            "Average Sphericity": otsu_avg_sphericity,
            "Average Aspect Ratio": otsu_avg_aspect_ratio
        },
        "Canny": {
            "Valid Particles": canny_valid_particles,
            "Total Particles": len(canny_contours),
            "Average Sphericity": canny_avg_sphericity,
            "Average Aspect Ratio": canny_avg_aspect_ratio
        }
    }

# Function to convert DataFrame to Excel
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Particle Analysis')
    processed_data = output.getvalue()
    return processed_data

# Main Streamlit App
st.sidebar.header("Input", divider='blue')
uploaded_files = st.sidebar.file_uploader("**Upload Particle Images**", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    all_data = []

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.subheader(f"Image {idx + 1}: {uploaded_file.name}", divider='blue')
        st.image(img_array, caption="Uploaded Image", use_column_width=True)

        features = process_image(img_array)

        # Store OTSU and Canny features in the list
        all_data.append({
            "Image Name": uploaded_file.name,
            "Filter Type": "OTSU",
            "Valid/Total Particles": f"{features['OTSU']['Valid Particles']}/{features['OTSU']['Total Particles']}",
            "Average Sphericity": features['OTSU']['Average Sphericity'],
            "Average Aspect Ratio": features['OTSU']['Average Aspect Ratio']
        })
        all_data.append({
            "Image Name": uploaded_file.name,
            "Filter Type": "Canny Edge",
            "Valid/Total Particles": f"{features['Canny']['Valid Particles']}/{features['Canny']['Total Particles']}",
            "Average Sphericity": features['Canny']['Average Sphericity'],
            "Average Aspect Ratio": features['Canny']['Average Aspect Ratio']
        })
    
    # Create DataFrame from all data
    df = pd.DataFrame(all_data)
    st.dataframe(df, use_container_width=True)

    # Convert DataFrame to Excel and provide download button
    excel_data = convert_df_to_excel(df)
    st.download_button(
        label="📥 Download Data as Excel",
        data=excel_data,
        file_name="particle_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
