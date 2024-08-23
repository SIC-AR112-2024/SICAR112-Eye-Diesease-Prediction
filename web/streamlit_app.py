import streamlit as st
import requests
from PIL import Image
import requests
from io import BytesIO


st.title("SIC AR112 - Identification of Eye Disease using retinal fundus images")

def home_page():
    st.title("Home Page")
    st.write("Welcome to the home page of our SIC Project - AR112!")
    st.write("The goal of this project is to test the effectiveness of Large Language Models (LLMs) at identifiying eye diseases from retinal fundus images in comparison to traditional machine learning (ML) models like ResNets.")

def classical_models_page():
    st.title("Classical Models")
    st.write("This is the about page.")

def zero_shot_model_page():
    st.title("0 - shot Model Page")
    st.write("This is the contact page.")
    
def chain_of_though_few_shot_model_page():
    st.title("Chain of thought few shot Model Page")
    st.write("This is the contact page.")

# Create a dictionary of subpages
subpages = {
    "Home": home_page,
    "Classical Models": classical_models_page,
    "0 - shot Model": zero_shot_model_page,
    "Chain of thought few shot Model": chain_of_though_few_shot_model_page
}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(subpages.keys()))

# Display the selected page
subpages[page]()

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