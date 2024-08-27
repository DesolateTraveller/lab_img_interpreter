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

@st.cache_data(ttl="2h")
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)                                     # Apply Gaussian blur to reduce noise
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Apply Otsu's thresholding
    #kernel = np.ones((3,3), np.uint8)
    #cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    #contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary_image

@st.cache_data(ttl="2h")
def calculate_diameters(contours):
    diameters = []
    for contour in contours:
        area = cv2.contourArea(contour)
        diameter = np.sqrt(4 * area / np.pi)
        diameters.append(diameter)
    return diameters

@st.cache_data(ttl="2h")
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

@st.cache_data(ttl="2h")
def calculate_gaps(binary_image):
    inverted_image = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gap_areas = [cv2.contourArea(contour) for contour in contours]
    if gap_areas:
        max_gap = max(gap_areas)
        min_gap = min(gap_areas)
    else:
        max_gap = min_gap = 0
    total_gap_area = sum(gap_areas)
    return total_gap_area, len(contours), max_gap, min_gap

@st.cache_data(ttl="2h")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])
#st.divider()

if uploaded_file is not None:

#---------------------------------------------------------------------------------------------------------------------------------
### Content
#---------------------------------------------------------------------------------------------------------------------------------

    tab1, tab2 = st.tabs(["**Information**","**Segmentation**"])

#---------------------------------------------------------------------------------------------------------------------------------
### Information
#---------------------------------------------------------------------------------------------------------------------------------
    
    with tab1:
    
        col1, col2 = st.columns((0.7,0.3))
        with col1:

            #st.subheader("Image", divider='blue')
            image = np.array(Image.open(uploaded_file))
            contours, binary_image = process_image(image)
            diameters = calculate_diameters(contours)
            output_image = draw_contours(image, contours, diameters)
            total_gap_area, gap_count, max_gap, min_gap = calculate_gaps(binary_image)
            st.image(output_image, caption=f"Detected Molecules: {len(diameters)}", use_column_width=True)

            with col2:

                #st.subheader("Statistics", divider='blue')
        
                df = pd.DataFrame(diameters, columns=["Diameter (px)"])
                max_diameter = df["Diameter (px)"].max()
                min_diameter = df["Diameter (px)"].min()

                df['Type'] = ['Max' if d == max_diameter else 'Min' if d == min_diameter else '' for d in df["Diameter (px)"]]
                df = df.sort_values(by="Diameter (px)", ascending=False).reset_index(drop=True)
        
                st.write("**Diameter Statistics:**")
                st.write(f"Maximum Diameter: **{max_diameter:.2f}** px")
                st.write(f"Minimum Diameter: **{min_diameter:.2f}** px")
                st.write(f"No of Molecules: **{df.shape[0]}**")

                st.divider()

                st.write("**Gap Statistics:**")
                st.write(f"Total Gap Area: **{total_gap_area:.2f}** px²")
                st.write(f"Total Number of Gaps: **{gap_count}**")
                st.write(f"Maximum Gap Area: **{max_gap:.2f}** px²")
                st.write(f"Minimum Gap Area: **{min_gap:.2f}** px²")

                st.divider()
                st.write("**Diameters of detected molecules (in pixels):**")
                st.dataframe(df.style
                     .highlight_max(subset=['Diameter (px)'], color='lightgreen')
                     .highlight_min(subset=['Diameter (px)'], color='lightcoral')
                     .format({'Diameter (px)': '{:.2f}'})
                     , use_container_width=True)
                
                st.divider()
                csv = convert_df_to_csv(df)
                st.download_button(label="Download data as CSV",data=csv,file_name='molecule_diameters.csv',mime='text/csv',)

