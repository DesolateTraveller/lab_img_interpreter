import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from skimage import color, filters, measure, morphology

def process_image(image):
    # Convert image to grayscale
    gray_image = color.rgb2gray(image)
    
    # Apply Gaussian filter to reduce noise
    blurred_image = filters.gaussian(gray_image, sigma=2.0)
    
    # Apply Otsu's thresholding
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image < thresh
    
    # Remove small objects and noise
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=100)
    
    # Label the connected regions in the binary image
    labeled_image = measure.label(cleaned_image)
    
    return labeled_image

def calculate_diameters(labeled_image):
    diameters = []
    properties = measure.regionprops(labeled_image)
    for prop in properties:
        # Calculate the equivalent diameter
        diameter = prop.equivalent_diameter
        diameters.append(diameter)
    return diameters, properties

def draw_contours(image, properties, diameters):
    output = Image.fromarray(image)
    draw_output = ImageDraw.Draw(output)
    for prop, diameter in zip(properties, diameters):
        y, x = prop.centroid
        radius = diameter / 2
        
        # Draw the circle
        draw_output.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)], 
            outline="green", width=2
        )
        
        # Draw the diameter text
        draw_output.text((x - 20, y - 20), f'{int(diameter)}px', fill="red")
    
    return output

def main():
    st.title("Molecule Counter and Diameter Measurement")

    uploaded_file = st.file_uploader("Upload an image of molecules", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = np.array(Image.open(uploaded_file))
        
        # Process the image to find contours
        labeled_image = process_image(image)
        
        # Calculate diameters of the molecules
        diameters, properties = calculate_diameters(labeled_image)
        
        # Draw contours and diameters on the image
        output_image = draw_contours(image, properties, diameters)
        
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
