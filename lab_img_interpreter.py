import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
import io

# Function to remove background
def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Function to find contours and analyze molecules
def analyze_molecules(image, min_size=100):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameters = []
    centers = []

    for cnt in contours:
        if cv2.contourArea(cnt) > min_size:
            # Get the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            diameter = radius * 2
            diameters.append(diameter)
            centers.append((int(x), int(y)))

    return diameters, centers, contours

# Function to segment molecules based on diameter
def segment_molecules(diameters, centers, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(diameters).reshape(-1, 1))
    clusters = kmeans.labels_
    return clusters

# Main function to display the Streamlit app
def main():
    st.title("Molecule Analysis and Segmentation")

    uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = np.array(Image.open(uploaded_file))
        original_image = image.copy()
        processed_image = remove_background(image)
        
        # Analyze molecules
        diameters, centers, contours = analyze_molecules(processed_image)
        
        # Segment molecules based on diameter
        clusters = segment_molecules(diameters, centers)
        
        # Create a dataframe for the results
        df = pd.DataFrame({
            "Diameter (px)": diameters,
            "Cluster": clusters
        })
        
        # Calculate the maximum and minimum diameters
        max_diameter = df["Diameter (px)"].max()
        min_diameter = df["Diameter (px)"].min()
        
        # Display the results
        st.write("**Diameter Statistics:**")
        st.write(pd.DataFrame({"Max Diameter (px)": [max_diameter], "Min Diameter (px)": [min_diameter]}))
        
        st.write("**Diameters and Clusters of Detected Molecules (in pixels):**")
        df_sorted = df.sort_values(by="Diameter (px)", ascending=False)
        st.dataframe(df_sorted.style.highlight_max(subset=['Diameter (px)'], color='lightgreen')
                               .highlight_min(subset=['Diameter (px)'], color='lightcoral'))

        # Draw contours and cluster on the image
        output_image = original_image.copy()
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 100:
                cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(output_image, f"{clusters[i]}", centers[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Display the segmented image
        st.image(output_image, caption="Segmented Molecules", use_column_width=True)
        
        # Add a download button for the dataframe
        csv = df_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Data as CSV", data=csv, file_name='molecule_data.csv', mime='text/csv')

if __name__ == "__main__":
    main()
