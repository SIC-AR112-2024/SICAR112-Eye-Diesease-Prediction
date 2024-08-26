import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("Setting the project")
st.subheader("Introduction to the dataset")
st.write("We utilised a dataset of retinal fundus images obtained from Kaggle (link found below). The dataset had a relatively balanced spread of about a 1000 images per condition for healthy, cataract, diabetic retinopathy and glaucoma.")
st.markdown("[Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)")
st.subheader("Explaining the Confusion Matrix")
st.write("The confusion matrix is the evaluative framework that we used for evaluating our models. It resembles a 4 by 4 square grid, with the true label (i.e. the disease the retinal fundus image is logging) on the vertical axis, and the predicted label (i.e. the disease predicted by the model) on the horizontal axis. Hence the squares along the diagonal line from the top left corner to the bottom right corner represent accurate predictions. An example of a confusion matrix is shown below:")
# Load image from URL
urlneg1 = "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Confusion%20Matrix%20Accuracy%20Guarantee/Guarantee.png"
response = requests.get(urlneg1)
# Check if the request was successful
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    # Display the image using Streamlit
    st.image(image, caption="This is the confusion matrix of our in house trained model.", use_column_width=True)
else:
    st.error(f"Failed to load image. Status code: {response.status_code}")  # Display an error message
st.subheader("Identifying Diseases")
st.write("From some research, we noted the following characteristics in the retinal fundus images that were usually present in the retinal fundus images of the diseases.")
# Bulleted list with bold words using markdown
st.markdown("""
- **Cataract:** The lens would usually appear cloudy or opaque in the retinal fundus image, causing the whole image to be blurry in general.
- **Diabetic Retinopathy:** Small red dots scattered along the retina in the fundus image are characteristic of tiny, bulging blood vessels in the eye called microaneurysms. Cotton wool white spots on the retinal image can also signify inflammation and retinal damage caused by the disease. In addition, bleeding may be observed in the retinal fundus image when the microaneurysms rupture, causing haemorrhages. Soft and hard exudates amy also be present on the image.
- **Glaucoma:** The image shows an enlarged optic cup, causing the optic nerve head to appear enlarged and cupped in shape. This results in a higher optic cup to disk ratio. This also causes the thinning of the neuroretinal rim as the thickness of the retinal nerve fibre layer decreases. In addition, the retinal fundus images may display retinal blood vessel asymmetry at the optic cup.
""")

st.write("Detailed annotated retinal fundus images are avaliable for each of the diseases below.")

urls = ["https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Diagnosis%20Images/Cataract%20Diagnosis.png",
        "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Diagnosis%20Images/Diabetic%20Retinopathy%20Diagnosis%201.png",
        "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Diagnosis%20Images/Diabetic%20Retinopathy%20Diagnosis%202.png",
        "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Diagnosis%20Images/Glaucoma%20Diagnosis.png"]

captions = ["Cataract Diagnosis Annotated Retinal Fundus Image", "Diabetic Retinopathy Diagnosis Annotated Retinal Fundus Image", "Diabetic Retinopathy Diagnosis Annotated Retinal Fundus Image", "Glaucoma Diagnosis Annotated Retinal Fundus Image"]

for i in range(len(urls)):
    response = requests.get(urls[i])
    # Check if the request was successful
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        # Display the image using Streamlit
        st.image(image, caption=captions[i], use_column_width=True)
    else:
        st.error(f"Failed to load image. Status code: {response.status_code}")  # Display an error message