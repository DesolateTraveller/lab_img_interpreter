#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
from sklearn.cluster import KMeans
#----------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------
import cv2
import openpyxl
from io import BytesIO
from PIL import Image, ImageDraw
from skimage import color, filters, measure, morphology
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Particle Image Analysis | v0.2",
                   layout="wide",
                   page_icon="🖼️",            
                   initial_sidebar_state="auto",)
#---------------------------------------
st.title(f""":rainbow[Particle Image Analysis]""")
st.markdown(
    '''
    Developed by : **:blue[E&PT - Digital Solutions]** | 
    prepared by : <a href="mailto:avijit.chakraborty@clariant.com">Avijit Chakraborty</a> |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''', 
    unsafe_allow_html=True)
#st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### knowledge 
#---------------------------------------------------------------------------------------------------------------------------------

stats_expander = st.expander("**Information**", expanded=False)
with stats_expander:
#with st.sidebar.popover("**Information**",help="Knowledge about the app", use_container_width=False):  
    st.markdown("""
    ### 🖼️ **Image Analysis and Classification Process**

    #### 1. ✨ **Image Preprocessing:**

    Image enhancement is performed in two stages to separate the background and detect particle edges:

    - **🟢 OTSU Thresholding:** Automatically binarizes the image by separating background from particles based on pixel intensity.
    - **🔵 Canny Edge Detection:** Detects the edges of particles by calculating intensity gradients to outline their contours.

    > The main goal of preprocessing is to extract **clear contours** of the particles, which will be analyzed in the next stage.

    ---

    #### 2. 📊 **Feature Extraction:**

    After preprocessing, the following key features are extracted from the image for classification:

    - **Aspect Ratio:**  
      The aspect ratio is calculated by fitting a minimal area rectangle around each detected particle contour using OpenCV’s `cv2.minAreaRect()` function.  
      - Formula:  
        \[
        \text{Aspect Ratio} = \frac{\min(\text{width}, \text{height})}{\max(\text{width}, \text{height})}
        \]
      - **Interpretation:**  
        - A ratio close to 1 → More circular particle.  
        - A lower ratio → Elongated or irregularly shaped particle.

    - **Sphericity:**  
      Sphericity measures how round the particle is.  
      - Formula:  
        \[
        \text{Sphericity} = \frac{4 \cdot \pi \cdot \text{Area}}{\text{Perimeter}^2}
        \]
      - **Interpretation:**  
        - Sphericity close to 1 → A perfectly round particle.  
        - Lower values → Irregularly shaped particles.

    ---

    #### 3. 📋 **Criteria for Classification:**

    #### **🟢 Good Image:**
    - Particles are mostly circular (aspect ratio close to 1).
    - High average sphericity indicating well-formed particles.
    - Few particles touch the edges; most contours are well-separated.
    - Particles are evenly distributed with minimal overlapping.

    #### **🔴 Bad Image:**
    - Particles have a low aspect ratio, indicating elongation or irregular shapes.
    - Low sphericity, suggesting poorly shaped or deformed particles.
    - Many particles overlap or touch the edges, making them incomplete.
    - The number of valid particles is lower due to overlap, noise, or incomplete shapes.

    ---

    #### 4. 🎯 **Final Classification:**

    - **Primary Criterion:**  
      The **aspect ratio** is the primary factor in determining if the image is **Good** or **Bad**.  
      - If the average aspect ratio > **0.85**, the image is classified as **Good**.  
      - If the average aspect ratio < **0.85**, the image is classified as **Bad**.

    - **Additional Factors:**  
      - Average sphericity.
      - Number of valid particles.

    ---

    #### 5. 📌 **Example:**

    - **A Good Image:**  
      Particles are circular or close to circular, with minimal overlap. The aspect ratio and sphericity are high, and most particles are valid.

    - **A Bad Image:**  
      Particles are elongated or irregularly shaped, with many overlapping or incomplete. Aspect ratio and sphericity are low, and many particles are invalid.
""")

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h") 
def process_image(img_array):
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)                                # Convert the image to grayscale
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)                                       # Apply Gaussian blur to reduce noise
      
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)                # OTSU Thresholding
    otsu_contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                # Find contours for OTSU Threshold
    img_with_otsu_contours = cv2.drawContours(img_array.copy(), otsu_contours, -1, (0, 255, 0), 2)          # Draw contours for OTSU
    
    otsu_valid_particles = 0                                                                                # Calculate valid particles and sphericity for OTSU
    otsu_total_particles = len(otsu_contours)
    otsu_sphericities = []
    otsu_aspect_ratios = []
            
    for contour in otsu_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        rect = cv2.minAreaRect(contour)                                                                     # Calculate aspect ratio of the minimum area rectangle
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect_ratio = min(width, height) / max(width, height)
            otsu_aspect_ratios.append(aspect_ratio)
        if perimeter > 0:
            sphericity = 4 * np.pi * (area / (perimeter ** 2))                                              # Sphericity formula
            otsu_sphericities.append(sphericity)
            otsu_valid_particles += 1                                                                       # Assuming all particles are valid for this example
            
        otsu_avg_sphericity = np.mean(otsu_sphericities) if otsu_sphericities else 0
        otsu_area_avg_sphericity = np.mean([s * area for s, area in zip(otsu_sphericities, [cv2.contourArea(c) for c in otsu_contours])]) if otsu_sphericities else 0
        otsu_avg_aspect_ratio = np.mean(otsu_aspect_ratios) if otsu_aspect_ratios else 0

    canny_edges = cv2.Canny(blurred, 80, 170)                                                               # Canny Edge Detection
    canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)             # Find contours for Canny Edge Detection
    img_with_canny_contours = cv2.drawContours(img_array.copy(), canny_contours, -1, (255, 0, 0), 2)        # Draw contours for Canny
    #st.image(img_with_canny_contours, caption="Canny Edge Filter: Detected Particles", use_column_width=True)
    
    canny_valid_particles = 0                                                                               # Calculate valid particles and sphericity for Canny Edge
    canny_total_particles = len(canny_contours)
    canny_sphericities = []
    canny_aspect_ratios = []
            
    for contour in canny_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        rect = cv2.minAreaRect(contour)                                                                     # Calculate aspect ratio of the minimum area rectangle
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect_ratio = min(width, height) / max(width, height)
            canny_aspect_ratios.append(aspect_ratio)
        if perimeter > 0:
            sphericity = 4 * np.pi * (area / (perimeter ** 2))                                              # Sphericity formula
            canny_sphericities.append(sphericity)
            canny_valid_particles += 1                                                                      # Assuming all particles are valid for this example
                        
        canny_avg_sphericity = np.mean(canny_sphericities) if canny_sphericities else 0
        canny_area_avg_sphericity = np.mean([s * area for s, area in zip(canny_sphericities, [cv2.contourArea(c) for c in canny_contours])]) if canny_sphericities else 0
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

@st.cache_data(ttl="2h")
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Particle Analysis')
    processed_data = output.getvalue()
    return processed_data
#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------
st.sidebar.header("Input", divider='blue')
uploaded_files = st.sidebar.file_uploader("**Upload a particle image**", type=["jpg", "png", "jpeg"],accept_multiple_files=True)
#---------------------------------------------------------------------------------------------------------------------------------
### Content
#---------------------------------------------------------------------------------------------------------------------------------
if uploaded_files:
    all_data = []

    tab1, tab2 = st.tabs([ "**Visualization**", "**Information**"])
    with tab1:
        
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns((0.6,0.4))
            with col1:

                #st.subheader("Input", divider='blue')
                st.subheader(f"Image {idx + 1}: {uploaded_file.name}", divider='blue')
                with st.container(height=900,border=True):

                    st.image(img_array, caption="Uploaded Image", use_column_width=True)

            with col2:

                st.subheader("OTSU Thresholding", divider='blue')
                with st.container(height=400,border=True):   
                
                    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)                                                # Convert the image to grayscale
                    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0) 
                    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)                # OTSU Thresholding
                    otsu_contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                # Find contours for OTSU Threshold
                    img_with_otsu_contours = cv2.drawContours(img_array.copy(), otsu_contours, -1, (0, 255, 0), 2)          # Draw contours for OTSU
                    st.image(img_with_otsu_contours, caption="OTSU Filter: Detected Particles", use_column_width=True) 
        
                st.subheader("Canny Edge Thresholding", divider='blue')
                with st.container(height=400,border=True): 

                    canny_edges = cv2.Canny(blurred, 80, 170)                                                               # Canny Edge Detection
                    canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)             # Find contours for Canny Edge Detection
                    img_with_canny_contours = cv2.drawContours(img_array.copy(), canny_contours, -1, (255, 0, 0), 2)        # Draw contours for Canny
                    st.image(img_with_canny_contours, caption="Canny Edge Filter: Detected Particles", use_column_width=True)

# -------------------------------------------------------------------------------------------------------    

            with tab2:

                features = process_image(img_array)

                all_data.append({
                    "Image Name": uploaded_file.name,
                    #"Filter Type": "OTSU",
                    "Valid/Total Particles(OTSU)": f"{features['OTSU']['Valid Particles']}/{features['OTSU']['Total Particles']}",
                    "Average Sphericity(OTSU)": features['OTSU']['Average Sphericity'],
                    "Average Aspect Ratio(OTSU)": features['OTSU']['Average Aspect Ratio'],
                    "Valid/Total Particles(CannyEdge)": f"{features['Canny']['Valid Particles']}/{features['Canny']['Total Particles']}",
                    "Average Sphericity(CannyEdge)": features['Canny']['Average Sphericity'],
                    "Average Aspect Ratio(CannyEdge)": features['Canny']['Average Aspect Ratio']
                    })

                df = pd.DataFrame(all_data)
                st.dataframe(df, use_container_width=True)

                excel_data = convert_df_to_excel(df)
                st.download_button(label="📥 Download Data as Excel",data=excel_data,file_name="particle_analysis.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f'analysis_{uploaded_file.name}')
