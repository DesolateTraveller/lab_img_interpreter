#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------
import cv2
from PIL import Image, ImageDraw
from skimage import color, filters, measure, morphology
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Lab Image Interpreter | v0.1",
                    layout="wide",
                    page_icon="🖼️",            
                    initial_sidebar_state="collapsed")
#----------------------------------------
st.title(f""":rainbow[Lab Image Interpreter]""")
st.markdown(
    '''
    Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( :envelope: [Email](mailto:avijit.mba18@gmail.com) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''',
    unsafe_allow_html=True)
st.info('**A lightweight image-processing streamlit app that interprets the laboratory and microsopic images**', icon="ℹ️")
#st.divider()
#----------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_diameters(contours):
    diameters = []
    for contour in contours:
        area = cv2.contourArea(contour)
        diameter = np.sqrt(4 * area / np.pi)
        diameters.append(diameter)
    return diameters

def draw_contours(image, contours, diameters):
    output_image = image.copy()
    for contour, diameter in zip(contours, diameters):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(output_image, f'{int(diameter)} px', (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return output_image

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])
st.divider()

if uploaded_file is not None:

    col1, col2 = st.columns((0.7,0.3))
    with col1:

        st.subheader("Image", divider='blue')
        image = np.array(Image.open(uploaded_file))
        contours = process_image(image)
        diameters = calculate_diameters(contours)
        output_image = draw_contours(image, contours, diameters)
        st.image(output_image, caption=f"Detected Molecules: {len(diameters)}", use_column_width=True)

        with col2:

            st.subheader("Information", divider='blue')
        
            df = pd.DataFrame(diameters, columns=["Diameter (px)"])
            max_diameter = df["Diameter (px)"].max()
            min_diameter = df["Diameter (px)"].min()

            df['Type'] = ['Max' if d == max_diameter else 'Min' if d == min_diameter else '' for d in df["Diameter (px)"]]
            df = df.sort_values(by="Diameter (px)", ascending=False).reset_index(drop=True)
        
            st.write("**Diameter Statistics:**")
            st.write(f"Maximum Diameter: **{max_diameter:.2f}** px")
            st.write(f"Minimum Diameter: **{min_diameter:.2f}** px")

            st.divider()
            st.write("**Diameters of detected molecules (in pixels):**")
            st.dataframe(df.style
                     .highlight_max(subset=['Diameter (px)'], color='lightgreen')
                     .highlight_min(subset=['Diameter (px)'], color='lightcoral')
                     .format({'Diameter (px)': '{:.2f}'})
                     , use_container_width=True)


