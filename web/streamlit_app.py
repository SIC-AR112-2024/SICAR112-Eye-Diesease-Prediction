import streamlit as st
import requests
from PIL import Image
import requests
from io import BytesIO


st.title("SIC AR112 - Identification of Eye Disease using retinal fundus images")
# Load image from URL
url = "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Confusion%20Matrix%20Accuracy%20Guarantee/Guarantee.png"
response = requests.get(url)
# Check if the request was successful
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))

    # Display the image using Streamlit
    st.image(image, caption="Image from URL", use_column_width=True)
else:
    st.error(f"Failed to load image. Status code: {response.status_code}")  # Display an error message