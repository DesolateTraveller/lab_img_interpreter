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
import time
import os
import tempfile
import pickle
from PIL import Image, ImageDraw
#----------------------------------------
from sklearn.cluster import KMeans
from collections import Counter
#----------------------------------------
import plotly.express as px
import plotly.graph_objects as go
#---------------------------------------------------------------------------------------------------------------------------------
### Title for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Particle Image Analysis | v0.1",
                    layout="wide",
                    page_icon="🖼️",            
                    initial_sidebar_state="collapsed")

#---------------------------------------------------------------------------------------------------------------------------------
### CSS
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
    <style>
    .centered-info {
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        font-size: 15px;
        color: #007BFF; 
        background-color: #FFFFFF; 
        border-radius: 5px;
        border: 1px solid #007BFF;
        margin: 0px;
        padding: 5px 10px;
    }
    </style>
""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title-large">Particle Image Analysis</div>
    <div class="title-small">Version : 0.1</div>
    """,
    unsafe_allow_html=True)

#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2025 | Created by : <span class="highlight">Avijit Chakraborty</span> | <a href="mailto:avijit.mba18@gmail.com"> 📩 </a></p> <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h")
def preprocess_image(image):
    preProcessParameter = {
        "maxStructureSize": 5,
        "minStructureSize": 3,
        "tweakContrastStretchingMax": 1.0,
        "preProcessingBlurSize": 3
    }
    maxStructureSize = preProcessParameter['maxStructureSize']
    minStructureSize = preProcessParameter['minStructureSize']
    tweakContrastStretchingMax = preProcessParameter['tweakContrastStretchingMax']
    preProcessingBlurSize = preProcessParameter['preProcessingBlurSize']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (maxStructureSize, maxStructureSize))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gmo = cv2.addWeighted(gray, 1.0, opened, -1.0, 0.0)
    minv, maxv = np.min(gmo), np.max(gmo)
    gmo = ((gmo - minv) * (255 * tweakContrastStretchingMax) / (maxv - minv)).astype(np.uint8)
    blurred = cv2.GaussianBlur(gmo, (preProcessingBlurSize, preProcessingBlurSize), 0)
    return blurred

#----------------------------------------
@st.cache_data(ttl="2h")
def ProcessSingleImageGradientMag(cv2, blurred):
    scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(scharr_x, scharr_y)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    edges = cv2.Canny(blurred, 100, 200)
    edgesCopy = edges.copy()
    contours, hierarchy = cv2.findContours(edgesCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return magnitude, edges, contours, hierarchy

#----------------------------------------
@st.cache_data(ttl="2h")
def apply_otsu_binarization(image):
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return otsu, contours

#----------------------------------------
@st.cache_data(ttl="2h")
def apply_canny_edge(image):
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return edges, contours

#----------------------------------------
#def combine_contours(otsu_contours, canny_contours):
    #combined_contours = otsu_contours + [c for c in canny_contours if c not in otsu_contours]
    #return combined_contours

@st.cache_data(ttl="2h")  
def combine_contours(otsu_contours, canny_contours):
    otsu_contours = list(otsu_contours)
    canny_contours = list(canny_contours)
    otsu_contours_set = {frozenset(map(tuple, c.reshape(-1, 2))) for c in otsu_contours}
    unique_canny_contours = [c for c in canny_contours if frozenset(map(tuple, c.reshape(-1, 2))) not in otsu_contours_set]
    return otsu_contours + unique_canny_contours

#----------------------------------------
@st.cache_data(ttl="2h")
def analyze_contours(contours):
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
    all_areas = [cv2.contourArea(c) for c in contours]
    valid_areas = [cv2.contourArea(c) for c in valid_contours]
    all_lengths = [cv2.arcLength(c, True) for c in contours]
    valid_lengths = [cv2.arcLength(c, True) for c in valid_contours]
    all_hulls = [cv2.convexHull(c) for c in contours]
    valid_hulls = [cv2.convexHull(c) for c in valid_contours]
    all_hull_areas = [cv2.contourArea(h) for h in all_hulls]
    valid_hull_areas = [cv2.contourArea(h) for h in valid_hulls]
    return {
        "Total Contours": len(contours),
        "Valid Contours": len(valid_contours),
        "Sum of All Areas": sum(all_areas) if all_areas else 0,
        "Sum of Valid Areas": sum(valid_areas) if valid_areas else 0,
        "Total Contour Length": sum(all_lengths) if all_lengths else 0,
        "Valid Contour Length": sum(valid_lengths) if valid_lengths else 0,
        "sum of All Hull Areas": sum(all_hull_areas) if all_hull_areas else 0,
        "sum of Valid Hull Areas": sum(valid_hull_areas) if valid_hull_areas else 0,
    }
    
#----------------------------------------
@st.cache_data(ttl="2h")
def calculate_metrics(contours):
    total_contours = len(contours)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
    valid_count = len(valid_contours)
    aspect_ratios = [cv2.boundingRect(cnt)[2] / max(cv2.boundingRect(cnt)[3], 1) for cnt in valid_contours]
    sphericities = [4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2) for cnt in valid_contours if cv2.arcLength(cnt, True) > 0]
    avg_sphericity = np.mean(sphericities) if sphericities else 0
    avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0
    area_avg_sphericity = np.average(sphericities, weights=[cv2.contourArea(cnt) for cnt in valid_contours]) if sphericities else 0
    area_avg_aspect_ratio = np.average(aspect_ratios, weights=[cv2.contourArea(cnt) for cnt in valid_contours]) if aspect_ratios else 0
    return valid_count, total_contours, avg_sphericity, area_avg_sphericity, avg_aspect_ratio, area_avg_aspect_ratio

#----------------------------------------
@st.cache_data(ttl="2h")
def analyze_image(image):
    if image is None:
        raise ValueError("Invalid image: The image could not be loaded. Please check the file format.")
    
    height, width, channels = image.shape
    processed = preprocess_image(image)
    magnitude, edges, edges_contours, hierarchy = ProcessSingleImageGradientMag(cv2, processed)
    
    otsu_img, otsu_contours = apply_otsu_binarization(processed)
    canny_img, canny_contours = apply_canny_edge(processed)
    combined_contours = combine_contours(otsu_contours, canny_contours)
    
    contour_data_otsu = analyze_contours(otsu_contours)
    contour_data_canny = analyze_contours(canny_contours)
    contour_data_combined = analyze_contours(combined_contours)
    
    otsu_metrics = calculate_metrics(otsu_contours)
    canny_metrics = calculate_metrics(canny_contours)
    combined_metrics = calculate_metrics(combined_contours)
    
    # Detect particles and classify shapes
    height, width, channels = image.shape
    contours = detect_particles(image)
    shape_counts = Counter()
    particle_records = []  # Store structured particle data

    for i, contour in enumerate(contours, start=1):
        shape = classify_shape(contour)
        shape_counts[shape] += 1
        x, y, w, h = cv2.boundingRect(contour)
        diameter = (w + h) / 2 if shape in ["Round", "Ellipse"] else None
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        particle_records.append({
            "Particle No.": i,
            "Shape": shape,
            "Width": w,
            "Height": h,
            "Diameter": diameter,
            "Area": area,
            "Perimeter": perimeter,
            "Solidity": solidity,
            "Circularity": circularity
        })

    particle_data_df = pd.DataFrame(particle_records)
    shape_counts_prefixed = {f"shape_{k.lower()}": v for k, v in shape_counts.items()}

    return {
        "Image Height": height,
        "Image Width": width,
        "Color Channels": channels,
        "Total Particles": len(contours),
        #"Particle Data": particle_data_df,  # Store as DataFrame instead of raw list
        #"Shape Counts": shape_counts,
        **shape_counts_prefixed,

        # OTSU Metrics
        "OTSU Valid/Total Contours": f"{otsu_metrics[0]}/{otsu_metrics[1]}",
        "OTSU Average Sphericity": otsu_metrics[2],
        "OTSU Area Averaged Sphericity": otsu_metrics[3],
        "OTSU Aspect Ratio": otsu_metrics[4],
        "OTSU Area Average Aspect Ratio": otsu_metrics[5],
        **{f"OTSU {k}": v for k, v in contour_data_otsu.items()},
                
        # Canny Metrics
        "Canny Valid/Total Contours": f"{canny_metrics[0]}/{canny_metrics[1]}",
        "Canny Average Sphericity": canny_metrics[2],
        "Canny Area Averaged Sphericity": canny_metrics[3],
        "Canny Aspect Ratio": canny_metrics[4],
        "Canny Area Average Aspect Ratio": canny_metrics[5],
        **{f"Canny {k}": v for k, v in contour_data_canny.items()},
        
        # Combined Metrics
        "Combined Valid/Total Contours": f"{combined_metrics[0]}/{combined_metrics[1]}",
        "Combined Average Sphericity": combined_metrics[2],
        "Combined Area Averaged Sphericity": combined_metrics[3],
        "Combined Aspect Ratio": combined_metrics[4],
        "Combined Area Average Aspect Ratio": combined_metrics[5],
        **{f"Combined {k}": v for k, v in contour_data_combined.items()},

    }
    
#----------------------------------------
@st.cache_data(ttl="2h")
def plot_metrics(image_results):
    fig, axes = plt.subplots(6, 2, figsize=(15, 22))
    metrics = [
        ("OTSU Average Sphericity", "OTSU Area Averaged Sphericity"),
        ("OTSU Aspect Ratio", "OTSU Area Average Aspect Ratio"),
        ("Canny Average Sphericity", "Canny Area Averaged Sphericity"),
        ("Canny Aspect Ratio", "Canny Area Average Aspect Ratio"),
        ("Combined Average Sphericity", "Combined Area Averaged Sphericity"),
        ("Combined Aspect Ratio", "Combined Area Average Aspect Ratio")
    ]
    def scatter_with_labels(ax, data, title, ylabel):
        for idx, value, label in data:
            color = 'blue' if 'good' in label.lower() else 'red'
            ax.scatter(idx, value, color=color)
            ax.annotate(f"{idx}", (idx, value), textcoords="offset points", xytext=(3, 3), ha='right', fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Image Index")
        ax.set_ylabel(ylabel)
    for row, (metric1, metric2) in enumerate(metrics):
        data1 = [(idx, result[metric1], result.get("Image", "").lower()) for idx, result in enumerate(image_results)]
        data2 = [(idx, result[metric2], result.get("Image", "").lower()) for idx, result in enumerate(image_results)]
        scatter_with_labels(axes[row, 0], data1, metric1, "Value")
        scatter_with_labels(axes[row, 1], data2, metric2, "Value")
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Good',
                            markerfacecolor='blue', markersize=8)
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Bad',
                           markerfacecolor='red', markersize=8)
    fig.legend(handles=[blue_patch, red_patch], loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for legend at top
    st.pyplot(fig, use_container_width=True)
    
#----------------------------------------------------------------------------------------------
@st.cache_data(ttl="2h")
def plot_selected_metrics_plotly(df, selected_columns):
    if not selected_columns:
        st.warning("Please select at least one metric to plot.")
        return
    for col in selected_columns:
        st.subheader(f"📈 {col}")     
        fig = go.Figure()
        for label_type, color in [("good", "blue"), ("bad", "red")]:
            filtered_df = df[df["Image"].str.lower().str.contains(label_type)]
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=filtered_df[col],
                mode='markers+text',
                text=filtered_df.index,
                textposition='top right',
                marker=dict(color=color, size=8),
                name=f"{label_type.title()} Images"
            ))
        fig.update_layout(
            xaxis_title="Image Index",
            yaxis_title=col,
            legend_title="Image Label",
            height=400,
            margin=dict(t=40, l=20, r=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
#----------------------------------------------------------------------------------------------
@st.cache_data(ttl="2h")
def detect_particles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

#----------------------------------------
@st.cache_data(ttl="2h")                                            # Classify shape based on aspect ratio and contour approximation.
def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif len(approx) == 5:
        return "Pentagon"
    elif len(approx) == 6:
        return "Hexagon"
    elif len(approx) > 6:
        return "Round" if 0.9 <= aspect_ratio <= 1.1 else "Ellipse"
    else:
        return "Other"
    
#----------------------------------------
@st.cache_data(ttl="2h")
def analyze_image_2(image):
    if image is None:
        raise ValueError("Invalid image: The image could not be loaded. Please check the file format.")
    height, width, _ = image.shape
    contours = detect_particles(image)
    particle_data = []
    shape_counts = Counter()
    for i, contour in enumerate(contours, start=1):
        shape = classify_shape(contour)
        shape_counts[shape] += 1
        x, y, w, h = cv2.boundingRect(contour)
        diameter = (w + h) / 2 if shape in ["Round", "Ellipse"] else None
        particle_data.append([i, shape, w, h, diameter])
    return width, height, len(contours), particle_data, shape_counts

#----------------------------------------
@st.cache_data
def analyze_image_cached(image_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name
        image = cv2.imread(temp_path)

    analysis = analyze_image(image)
    width, height, num_particles, particle_data, shape_counts = analyze_image_2(image)
    return analysis, width, height, num_particles, particle_data, shape_counts

#----------------------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

#----------------------------------------
@st.cache_resource
def save_plot(fig):
    path = os.path.join(tempfile.gettempdir(), "all_metrics.png")
    fig.savefig(path, bbox_inches="tight")
    return path

#----------------------------------------
def calculate_image_features(image):
        processed = preprocess_image(image)
        contours = detect_particles(image)

        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        aspect_ratios = [cv2.boundingRect(cnt)[2] / max(cv2.boundingRect(cnt)[3], 1) for cnt in valid_contours]
        sphericities = [4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2)
                        for cnt in valid_contours if cv2.arcLength(cnt, True) > 0]

        features_dict = {
            "Image Height": image.shape[0],
            "Image Width": image.shape[1],
            "Total Particles": len(contours),
            "Average Sphericity": np.mean(sphericities) if sphericities else 0,
            "Area Averaged Sphericity": np.average(sphericities, weights=[cv2.contourArea(cnt) for cnt in valid_contours]) if sphericities else 0,
            "Aspect Ratio": np.mean(aspect_ratios) if aspect_ratios else 0,
            "Area Average Aspect Ratio": np.average(aspect_ratios, weights=[cv2.contourArea(cnt) for cnt in valid_contours]) if aspect_ratios else 0
        }

        shape_counts = Counter([classify_shape(cnt) for cnt in valid_contours])
        for shape in ["Round", "Ellipse", "Square", "Rectangle", "Triangle", "Other"]:
            features_dict[f"shape_{shape.lower()}"] = shape_counts.get(shape, 0)

        return pd.DataFrame([features_dict])
   

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------
with st.popover("**:red[App Capabilities]**", disabled=False, use_container_width=True): 
    with st.container(border=True):
     
        st.info("""
                
                **Image Preprocessing has three stages : Image Enhancement, Image Extraction & Image Assessment**
                
                   """)  
        
        col1, col2, col3 = st.columns((0.3,0.4,0.3))
        with col1:
            
            st.info("""
                    
                    ✨ **Image Enhancement**:
                    
                    * It aims at increasing contrasts at the particle boundaries as much as possible so that a subsequent binarization unambiguously extracts the particle boundary contours. 
                    * It is performed in two stages to separate the background and detect particle edges:

                    🟢 **OTSU Thresholding**: Automatically binarizes the image by separating background from particles based on pixel intensity.
                    
                    🔵 **Canny Edge Detection**: Detects the edges of particles by calculating intensity gradients to outline their contours.

                    - The main goal of preprocessing is to extract **clear contours** of the particles, which will be analyzed in the next stage.
                    
                    """) 
              
        with col2:
            
            st.info("""
            
                    📊 **Image Extraction**:
                    
                    After enhancing, the following key features are extracted from the image for classification:

                    **Aspect Ratio**:
                    * Measures the particle's shape, particularly its elongation or flatness.
                    * Formula: `Aspect Ratio = min(width, height) / max(width, height)`
                    * Interpretation:
                        - A ratio close to 1 indicates a more circular particle.
                        - A lower ratio indicates an elongated or irregularly shaped particle.

                    **Sphericity**:
                    * Measures how round the particle is.
                    * Formula: `Sphericity = (4 * pi * Area) / Perimeter^2`
                    * Interpretation:
                        - Sphericity close to 1 indicates a perfectly round particle.
                        - Lower values indicate irregularly shaped particles.
                    
                    """) 
              
        with col3:
            
            st.info("""
            
                    📋 **Image Assessment**:
                    
                    * In this stage, based on the extracted information, we can asses type of image **Good/Bad**.
            
                    🟢 **Good Image**:
                    * Particles are mostly circular (aspect ratio close to 1).
                    * High average sphericity indicating well-formed particles.
                    * Few particles touch the edges; most contours are well-separated.
                    * Particles are evenly distributed with minimal overlapping.

                    🔴 **Bad Image**:
                    * Particles have a low aspect ratio, indicating elongation or irregular shapes.
                    * Low sphericity, suggesting poorly shaped or deformed particles.
                    * Many particles overlap or touch the edges, making them incomplete.
                    *  The number of valid particles is lower due to overlap, noise, or incomplete shapes.
                    
                    """) 
            
        st.info("""
                    **Additional Notes:**

                    * The preprocessing steps are crucial for ensuring accurate particle detection and feature extraction.
                    * The aspect ratio and sphericity provide valuable insights into the shape and quality of the particles.
                    * The classification criteria help identify good and bad images based on the extracted features.

                    **Conclusion:**
                    
                    This report presents a comprehensive analysis of image particles, including preprocessing, feature extraction, classification, and examples. The findings can be used to assess the quality of the images and identify potential issues for further investigation.
                """)
    
#---------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns((0.2,0.8))
with col1:
    with st.container(border=True):      
    
        uploaded_files = st.file_uploader("**:blue[Choose Images]**", type=["jpg", "png", "jpeg", "tiff"], accept_multiple_files=True)
        if uploaded_files:
    
            with col2: 
    
                if "analysis_results" not in st.session_state:
                    st.session_state.analysis_results = []
                    st.session_state.image_info_df = pd.DataFrame()

                    for uploaded_file in uploaded_files:
                        image_bytes = uploaded_file.read()
                        analysis, width, height, num_particles, particle_data, shape_counts = analyze_image_cached(image_bytes, uploaded_file.name)

                        analysis["Image"] = uploaded_file.name
                        st.session_state.analysis_results.append(analysis)

                        st.session_state.image_info_df = pd.DataFrame(st.session_state.analysis_results)
                        st.session_state.image_info_df = st.session_state.image_info_df[["Image"] + [col for col in st.session_state.image_info_df.columns if col != "Image"]]
        
                        #------------------------------------------------------------------------
                        tab1, tab2 = st.tabs(["**Images**", "**Information**"])
                        #------------------------------------------------------------------------
                        with tab1:
                            with st.container(border=True):
        
                                for uploaded_file in uploaded_files:
            
                                    col1, col2 = st.columns((0.4,0.6))
                                    with col1:
                        
                                        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Original Image</span></div>',unsafe_allow_html=True,)
                                        with st.container(border=True):
                            
                                            image = Image.open(uploaded_file)
                                            img_array = np.array(image)
                                            st.image(img_array, caption=f"Image: {uploaded_file.name}")
                    
                                    with col2:                     
                                    
                                        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Contoured Image</span></div>',unsafe_allow_html=True,)       
                                         
                                        subcol1,subcol2 = st.columns(2)
                                        with subcol1:
                                            with st.container(border=True):
                        
                                                gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)                                                # Convert the image to grayscale
                                                blurred = cv2.GaussianBlur(gray_image, (5, 5), 0) 
                                                _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)                # OTSU Thresholding
                                                otsu_contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                # Find contours for OTSU Threshold
                                                img_with_otsu_contours = cv2.drawContours(img_array.copy(), otsu_contours, -1, (0, 255, 0), 2)          # Draw contours for OTSU 
                                                st.image(img_with_otsu_contours, caption="OTSU Filter: Detected Particles")
                        
                                        with subcol2:
                                            with st.container(border=True):                        
                        
                                                canny_edges = cv2.Canny(blurred, 80, 170)                                                               # Canny Edge Detection
                                                canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)             # Find contours for Canny Edge Detection
                                                img_with_canny_contours = cv2.drawContours(img_array.copy(), canny_contours, -1, (255, 0, 0), 2)        # Draw contours for Canny
                                                st.image(img_with_canny_contours, caption="Canny Edge Filter: Detected Particles")    
                    
                                    st.write('------------------')  
                                                        
                        #------------------------------------------------------------------------              
                        with tab2:
                            with st.container(border=True):
            
                                st.table(st.session_state.image_info_df)
                                st.sidebar.divider()
                                csv_data = convert_df_to_csv(st.session_state.image_info_df)
                                st.sidebar.download_button(label="**:blue[📥  Download | Image Information (.csv)]**",data=csv_data,file_name="particle_image_analysis.csv",mime="text/csv",key="csv_download") 
                    
                        

