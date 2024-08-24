import streamlit as st
from PIL import Image
st.title("Applets")
st.write("Trialing photo upload feature.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the image in the app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Optionally, you can process the image here
    st.write("Image successfully uploaded and displayed!")