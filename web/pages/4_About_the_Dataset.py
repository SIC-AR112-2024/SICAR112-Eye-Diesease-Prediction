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