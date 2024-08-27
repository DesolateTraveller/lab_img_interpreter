import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def process_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, binary_image

def calculate_diameters(contours):
    diameters = []
    for contour in contours:
        # Calculate the equivalent diameter
        area = cv2.contourArea(contour)
        diameter = np.sqrt(4 * area / np.pi)
        diameters.append(diameter)
    return diameters

def draw_contours(image, contours, diameters):
    output = image.copy()
    for contour, diameter in zip(contours, diameters):
        x, y, w, h = cv2.boundingRect(contour)
        radius = int(diameter / 2)
        
        # Draw the circle around the contour
        cv2.circle(output, (x + w // 2, y + h // 2), radius, (0, 255, 0), 2)
        
        # Draw the diameter text
        cv2.putText(output, f'{int(diameter)}px', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    return output

def calculate_gaps(binary_image):
    # Invert binary image to detect gaps
    inverted_image = cv2.bitwise_not(binary_image)
    
    # Find contours of the gaps
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    gap_areas = [cv2.contourArea(contour) for contour in contours]
    total_gap_area = sum(gap_areas)
    
    return total_gap_area, len(contours)

def main():
    st.title("Molecule Analyzer")

    uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = np.array(Image.open(uploaded_file))
        
        # Process the image to find contours
        contours, binary_image = process_image(image)
        
        # Calculate diameters of the molecules
        diameters = calculate_diameters(contours)
        
        # Draw contours and diameters on the image
        output_image = draw_contours(image, contours, diameters)
        
        # Calculate gap area and count gaps
        total_gap_area, gap_count = calculate_gaps(binary_image)
        
        # Display the processed image with contours
        st.image(output_image, caption=f"Detected Molecules: {len(diameters)}", use_column_width=True)
        
        # Convert diameters list to a DataFrame
        df = pd.DataFrame(diameters, columns=["Diameter (px)"])
        
        # Calculate and display maximum and minimum diameters
        max_diameter = df["Diameter (px)"].max()
        min_diameter = df["Diameter (px)"].min()
        
        # Add a type column to mark max and min values
        df['Type'] = ['Max' if d == max_diameter else 'Min' if d == min_diameter else '' for d in df["Diameter (px)"]]
        
        # Reorder the DataFrame to put max and min at the top
        df = df.sort_values(by='Type', ascending=False).reset_index(drop=True)
        
        # Display the DataFrame as a table
        st.write("**Diameters of detected molecules (in pixels):**")
        st.dataframe(df.style.highlight_max(subset=['Diameter (px)'], color='lightgreen').highlight_min(subset=['Diameter (px)'], color='lightcoral').set_properties(**{'text-align': 'center'}), width=1000, height=600)
        
        # Display the total gap area and count of gaps
        st.write(f"**Total Gap Area:** {total_gap_area:.2f} px²")
        st.write(f"**Total Number of Gaps:** {gap_count}")

if __name__ == "__main__":
    main()
