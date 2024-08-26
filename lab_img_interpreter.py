import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw

def process_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small objects and noise using morphological operations
    kernel = np.ones((3,3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours of the objects
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def calculate_diameters(contours):
    diameters = []
    for contour in contours:
        # Calculate the equivalent diameter
        area = cv2.contourArea(contour)
        diameter = np.sqrt(4 * area / np.pi)
        diameters.append(diameter)
    return diameters

def draw_contours(image, contours, diameters):
    output_image = image.copy()
    for contour, diameter in zip(contours, diameters):
        # Get the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        # Draw the contour
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        
        # Draw the diameter text
        cv2.putText(output_image, f'{int(diameter)} px', (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return output_image

def main():
    st.title("Molecule Counter and Diameter Measurement with OpenCV")
    
    # Load the image
    uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        
        # Process the image to find contours
        contours = process_image(image)
        
        # Calculate diameters of the molecules
        diameters = calculate_diameters(contours)
        
        # Draw contours and diameters on the image
        output_image = draw_contours(image, contours, diameters)
        
        # Display the processed image with contours
        st.image(output_image, caption=f"Detected Molecules: {len(diameters)}", use_column_width=True)
        
        # Convert diameters list to a DataFrame
        df = pd.DataFrame(diameters, columns=["Diameter (px)"])
        
        # Calculate and display maximum and minimum diameters
        max_diameter = df["Diameter (px)"].max()
        min_diameter = df["Diameter (px)"].min()
        
        st.write("**Diameter Statistics:**")
        st.write(f"Maximum Diameter: {max_diameter:.2f} px")
        st.write(f"Minimum Diameter: {min_diameter:.2f} px")
        
        # Highlight the row with maximum and minimum diameters in the table
        df['Type'] = ['Max' if d == max_diameter else 'Min' if d == min_diameter else '' for d in df["Diameter (px)"]]
        
        # Display the DataFrame as a table
        st.write("**Diameters of detected molecules (in pixels):**")
        st.dataframe(df.style.highlight_max(subset=['Diameter (px)'], color='lightgreen').highlight_min(subset=['Diameter (px)'], color='lightcoral'))

if __name__ == "__main__":
    main()
