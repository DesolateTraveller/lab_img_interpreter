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
    gray_image = color.rgb2gray(image)
    blurred_image = filters.gaussian(gray_image, sigma=2.0)
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image < thresh
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=100)
    labeled_image = measure.label(cleaned_image)
    return labeled_image

@st.cache_data(ttl="2h")
def calculate_diameters(labeled_image):
    diameters = []
    properties = measure.regionprops(labeled_image)
    for prop in properties:
        diameter = prop.equivalent_diameter
        diameters.append(diameter)
    return diameters, properties

@st.cache_data(ttl="2h")
def draw_contours(image, properties, diameters):
    output = Image.fromarray(image)
    draw_output = ImageDraw.Draw(output)
    for prop, diameter in zip(properties, diameters):
        y, x = prop.centroid
        radius = diameter / 2
        draw_output.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)], 
            outline="green", width=2
        )
        draw_output.text((x - 20, y - 20), f'{int(diameter)}px', fill="red")
    return output

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])
st.divider()

if uploaded_file is not None:

    col1, col2 = st.columns((0.7,0.3))
    with col1:

        image = np.array(Image.open(uploaded_file))
        labeled_image = process_image(image)
        diameters, properties = calculate_diameters(labeled_image)
        output_image = draw_contours(image, properties, diameters)
        st.image(output_image, caption=f"Detected Molecules: {len(diameters)}", use_column_width=True)

        with col2:

            st.subheader("Information", divider='blue')
        
            df = pd.DataFrame(diameters, columns=["Diameter (px)"])
            max_diameter = df["Diameter (px)"].max()
            min_diameter = df["Diameter (px)"].min()
            st.write("**Diameter Statistics:**")
            st.write(f"Maximum Diameter: {max_diameter:.2f} px")
            st.write(f"Minimum Diameter: {min_diameter:.2f} px")
        
            st.divider()

            df['Type'] = ['Max' if d == max_diameter else 'Min' if d == min_diameter else '' for d in df["Diameter (px)"]]
            st.write("**Diameters of detected molecules (in pixels):**")
            st.dataframe(df.style.highlight_max(subset=['Diameter (px)'], color='lightgreen').highlight_min(subset=['Diameter (px)'], color='lightcoral'))

