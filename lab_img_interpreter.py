import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage import color, measure, morphology

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

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
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
def draw_contours(image, contours, diameters, clusters, cluster_counts):
    output_image = image.copy()
    cluster_centers = {}

    for i, (contour, diameter, cluster) in enumerate(zip(contours, diameters, clusters)):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(output_image, f'{int(diameter)} px (C{cluster})', (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if cluster not in cluster_centers:
            cluster_centers[cluster] = (cX, cY, 1)
        else:
            cX_sum, cY_sum, count = cluster_centers[cluster]
            cluster_centers[cluster] = (cX_sum + cX, cY_sum + cY, count + 1)

    # Draw the cluster number and count of molecules in the center of each cluster
    for cluster, (cX_sum, cY_sum, count) in cluster_centers.items():
        cX_center = int(cX_sum / count)
        cY_center = int(cY_sum / count)
        cv2.putText(output_image, f'C{cluster}: {cluster_counts[cluster]} mol', 
                    (cX_center - 40, cY_center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
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
def segment_molecules(diameters, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(diameters).reshape(-1, 1))
    clusters = kmeans.labels_
    return clusters

@st.cache_data(ttl="2h")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    tab1, tab2 = st.tabs(["**Information**","**Segmentation**"])

    with tab1:
        col1, col2 = st.columns((0.7,0.3))
        with col1:
            image = np.array(Image.open(uploaded_file))
            masked_image, mask = remove_background(image)
            contours, binary_image = process_image(masked_image)
            diameters = calculate_diameters(contours)
            st.image(masked_image, caption=f"**Detected Molecules: {len(diameters)}**", use_column_width=True)

            with col2:
                df = pd.DataFrame(diameters, columns=["Diameter (px)"])
                max_diameter = df["Diameter (px)"].max()
                min_diameter = df["Diameter (px)"].min()

                df['Type'] = ['Max' if d == max_diameter else 'Min' if d == min_diameter else '' for d in df["Diameter (px)"]]
                df = df.sort_values(by="Diameter (px)", ascending=False).reset_index(drop=True)

                st.write("**Diameter Statistics:**")
                st.write(pd.DataFrame({"Max Diameter (px)": [max_diameter], "Min Diameter (px)":[min_diameter], "No of Molecules": [df.shape[0]]}))

                total_gap_area, gap_count, max_gap, min_gap = calculate_gaps(binary_image)

                st.divider()
                st.write("**Gap Statistics:**")
                gap_df = pd.DataFrame({
                    "Total Gap Area (px²)": [total_gap_area],
                    "Total Number of Gaps": [gap_count],
                    "Maximum Gap Area (px²)": [max_gap],
                    "Minimum Gap Area (px²)": [min_gap]
                })
                st.write(gap_df)

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

    with tab2:
        col1, col2 = st.columns((0.7,0.3))
        with col1:
            st.image(masked_image, caption="Masked Image", use_column_width=True)

            with col2:
                num_clusters = st.slider("**Select no of clusters**", 1, 10, 5, 1)
                clusters = segment_molecules(diameters, num_clusters)

                # Count the number of molecules in each cluster
                cluster_counts = pd.Series(clusters).value_counts().sort_index().to_dict()

                clustered_image = draw_contours(masked_image, contours, diameters, clusters, cluster_counts)
                st.image(clustered_image, caption="Clustered Molecules", use_column_width=True)

                st.divider()

                st.write("**Cluster Statistics:**")
                df_c = pd.DataFrame({"Diameter (px)": diameters, "Cluster": clusters})
                cluster_stats = df_c.groupby('Cluster').agg({'Diameter (px)': ['max', 'min', 'count']})
                cluster_stats.columns = ['Max Diameter (px)', 'Min Diameter (px)', 'Number of Molecules']
                st.dataframe(cluster_stats, use_container_width=True)

                st.divider()
                csv_clusters = convert_df_to_csv(df_c)
                st.download_button(label="Download cluster data as CSV",data=csv_clusters,file_name='molecule_clusters.csv',mime='text/csv',)
