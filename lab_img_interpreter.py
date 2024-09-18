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
from PIL import Image, ImageDraw
from skimage import color, filters, measure, morphology
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Particle Image Analysis | v0.1",
                    layout="wide",
                    page_icon="🖼️",            
                    initial_sidebar_state="auto",)
#----------------------------------------
st.title(f""":rainbow[Particle Image Analysis]""")
st.markdown(
    '''
    Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( :envelope: [Email](mailto:avijit.mba18@gmail.com) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''',
    unsafe_allow_html=True)
st.info('**A lightweight image-processing streamlit app that interprets the laboratory and microsopic images**', icon="ℹ️")
st.divider()
#----------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h")
def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

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

@st.cache_data(ttl="2h")
def segment_molecules(diameters, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(diameters).reshape(-1, 1))
    clusters = kmeans.labels_
    return clusters
#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------
st.sidebar.header("Input", divider='blue')
uploaded_file = st.sidebar.file_uploader("**Upload a particle image**",type=["jpg", "jpeg", "png"])
#st.divider()

if uploaded_file is not None:

#---------------------------------------------------------------------------------------------------------------------------------
### Content
#---------------------------------------------------------------------------------------------------------------------------------

    tab1, tab2 = st.tabs(["**Input**","**Information**"])

#---------------------------------------------------------------------------------------------------------------------------------
### Input
#---------------------------------------------------------------------------------------------------------------------------------
    
    with tab1:
    
        col1, col2 = st.columns(2)
        with col1:

            st.subheader("Input", divider='blue')
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(img_array, caption="Uploaded Image", use_column_width=True)

        with col2:
                
            st.subheader("Molecule Detection", divider='blue')
            image = np.array(Image.open(uploaded_file))
            contours, binary_image = process_image(image)
            diameters = calculate_diameters(contours)
            output_image = draw_contours(image, contours, diameters)
            total_gap_area, gap_count, max_gap, min_gap = calculate_gaps(binary_image)
            st.image(output_image, caption=f"Detected Molecules: {len(diameters)}", use_column_width=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Information
#---------------------------------------------------------------------------------------------------------------------------------
    
    with tab2:

        col1, col2, col3 = st.columns((0.3,0.3,0.4))
        with col1:

                st.subheader("Statistics", divider='blue')
        
                df = pd.DataFrame(diameters, columns=["Diameter (px)"])
                max_diameter = df["Diameter (px)"].max()
                min_diameter = df["Diameter (px)"].min()

                df['Type'] = ['Max' if d == max_diameter else 'Min' if d == min_diameter else '' for d in df["Diameter (px)"]]
                df = df.sort_values(by="Diameter (px)", ascending=False).reset_index(drop=True)
        
                st.write("**Diameter Statistics:**")
                dia_stats_df = pd.DataFrame({
                    "Metric": ["Max Diameter (px)", "Min Diameter (px)", "No of Molecules"],
                    "Value": [max_diameter, min_diameter, df.shape[0]]
                })
                st.dataframe(dia_stats_df, use_container_width=True)
              
                #st.write(pd.DataFrame({"Max Diameter (px)": [max_diameter], "Min Diameter (px)": [min_diameter], "No of Molecules": [df.shape[0]]} ))
                
                st.divider()

                st.write("**Gap Statistics:**")
                gap_stats_df = pd.DataFrame({
                    "Metric": ["Total Gap Area (px²)", "Total Number of Gaps", "Maximum Gap Area (px²)", "Minimum Gap Area (px²)"],
                    "Value": [total_gap_area, gap_count, max_gap, min_gap]
                })
                st.dataframe(gap_stats_df, use_container_width=True)

                st.divider()
                st.write("**Diameters of detected molecules (in pixels):**")
                st.dataframe(df.style
                     .highlight_max(subset=['Diameter (px)'], color='lightgreen')
                     .highlight_min(subset=['Diameter (px)'], color='lightcoral')
                     .format({'Diameter (px)': '{:.2f}'})
                     , use_container_width=True)
                
                st.sidebar.divider()
                csv = convert_df_to_csv(df)
                #st.sidebar.download_button(label="Download molecule statistics as CSV",data=csv,file_name='molecule_diameters.csv',mime='text/csv',)

            #masked_image, mask = remove_background(image)
            #st.image(masked_image,caption="Masked Image", use_column_width=True)

        with col2:

                num_clusters = st.sidebar.slider("**Select no of clusters**", 1, 10, 5, 1)
                clusters = segment_molecules(diameters, num_clusters)

                st.subheader("Clusters", divider='blue')
                df_c = pd.DataFrame({"Diameter (px)": diameters,"Cluster": clusters})
                cluster_stat = df_c.groupby("Cluster").agg(Max_Diameter =("Diameter (px)", "max"),
                                                           Min_Diameter =("Diameter (px)", "min"),
                                                           Mean_Diameter =("Diameter (px)", "mean"),).reset_index()                
                st.dataframe(cluster_stat, use_container_width=True)

                cluster_no = df_c.groupby('Cluster').agg({'Diameter (px)': ['count']})
                cluster_no.columns = ['Number of Molecules']
                st.dataframe(cluster_no, use_container_width=True)

                #st.divider()
                csv_clusters = convert_df_to_csv(df_c)

                st.sidebar.download_button(label="Download molecule statistics as CSV",data=csv,file_name='molecule_diameters.csv',mime='text/csv',)
                st.sidebar.download_button(label="Download cluster data as CSV",data=csv_clusters,file_name='molecule_clusters.csv',mime='text/csv',)

        with col3:

            gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)                                # Convert the image to grayscale
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)                                       # Apply Gaussian blur to reduce noise

            # -------------------------------------------------------------------------------------------------------
            #stats_expander = st.expander("**:red[OTSU Thresholding]**", expanded=False)
            #with stats_expander:
            st.subheader("OTSU Thresholding", divider='blue')
            with st.container(height=400,border=True):    
                        
                    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)                # OTSU Thresholding
                    otsu_contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                # Find contours for OTSU Threshold
                    img_with_otsu_contours = cv2.drawContours(img_array.copy(), otsu_contours, -1, (0, 255, 0), 2)          # Draw contours for OTSU
                    st.image(img_with_otsu_contours, caption="OTSU Filter: Detected Particles", use_column_width=True)
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
            # -------------------------------------------------------------------------------------------------------
            #stats_expander = st.expander("**:red[Canny Edge Thresholding]**", expanded=False)
            #with stats_expander:
            st.subheader("Canny Edge Thresholding", divider='blue')
            with st.container(height=400,border=True):  
                     
                    canny_edges = cv2.Canny(blurred, 80, 170)                                                               # Canny Edge Detection
                    canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)             # Find contours for Canny Edge Detection
                    img_with_canny_contours = cv2.drawContours(img_array.copy(), canny_contours, -1, (255, 0, 0), 2)        # Draw contours for Canny
                    st.image(img_with_canny_contours, caption="Canny Edge Filter: Detected Particles", use_column_width=True)
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
            # -------------------------------------------------------------------------------------------------------    
            #st.divider()
            data = {"Filter Type": ["OTSU", "Canny Edge"],
                    "Valid/Total Particles": [f"{otsu_valid_particles}/{otsu_total_particles}", f"{canny_valid_particles}/{canny_total_particles}"],
                    "Average Sphericity": [otsu_avg_sphericity, canny_avg_sphericity],
                    "Area Averaged Sphericity": [otsu_area_avg_sphericity, canny_area_avg_sphericity],
                    "Average Aspect Ratio": [otsu_avg_aspect_ratio, canny_avg_aspect_ratio],}
            df = pd.DataFrame(data)
            st.dataframe(df)

    # -------------------------------------------------------------------------------------------------------

    st.sidebar.divider()
    st.sidebar.header("Hyperparameters", divider='blue')

    aspect_ratio_threshold = st.sidebar.slider("**aspect_ratio_threshold**", min_value=0.01, max_value=0.99, value=0.8)
    sphericity_threshold = st.sidebar.slider("**sphericity_threshold**", min_value=0.01, max_value=0.99, value=0.7)
    particle_ratio_threshold = st.sidebar.slider("**particle_ratio_threshold**", min_value=0.01, max_value=0.99, value=0.5)

    st.sidebar.divider()

    if otsu_avg_aspect_ratio > aspect_ratio_threshold and otsu_avg_sphericity > sphericity_threshold and (otsu_valid_particles / otsu_total_particles) > particle_ratio_threshold:
        st.sidebar.success('''
                           Based on above hyperparameters, the image considered as :
                           **GOOD**
                           ''')
    elif canny_avg_aspect_ratio > aspect_ratio_threshold and canny_avg_sphericity > sphericity_threshold and (canny_valid_particles / canny_total_particles) > particle_ratio_threshold:
        st.sidebar.success('''
                           Based on above hyperparameters, the image considered as :
                           **GOOD**
                           ''')
    else:
        st.sidebar.error('''
                        Based on above hyperparameters, the image considered as :
                        **BAD**
                        ''')
