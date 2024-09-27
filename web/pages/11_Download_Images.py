import streamlit as st
from PIL import Image
import io

# Load or create an image
image = Image.open('web/more_images/Logo_Image.png')  # Replace with your image file path or create an image dynamically

if image.mode == 'RGBA':
    image = image.convert('RGB')  # Convert to RGB (no alpha)

# Convert the image to bytes for download
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='JPEG')  # You can change format if needed (PNG, etc.)
img_byte_arr = img_byte_arr.getvalue()

# Display the image in Streamlit
st.image(image, caption="Sample Image", use_column_width=True)

# Provide a download button for the image
st.download_button(
    label="Download Image",
    data=img_byte_arr,
    file_name="downloaded_image.jpg",  # The name of the file to be downloaded
    mime="image/jpeg"  # Adjust mime type depending on the image format
)