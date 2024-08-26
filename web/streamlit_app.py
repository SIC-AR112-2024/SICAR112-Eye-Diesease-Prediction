import streamlit as st
import requests
from PIL import Image
import requests
from io import BytesIO
#commit test

st.title("SIC AR112 - Identification of Eye Disease using retinal fundus images")

def home_page():
    st.title("Home Page")
    st.write("Welcome to the home page of our SIC Project - AR112!")
    st.write("The goal of this project is to test the effectiveness of Large Language Models (LLMs) at identifiying eye diseases from retinal fundus images in comparison to traditional machine learning (ML) models like ResNets.")

def dataset_and_confusion_matrix_page():
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

def classical_models_page():
    st.title("Classical Models")
    st.subheader("Results")
    st.write("Firstly, we tried to analyse the retinal fundus images using a few models. These include the PyTorch pretrained resnet18, resnet34 and resnet50 models, our own customised resnet18 model, a Convolutional Neural Network model that we built, as well as a GoogLeNet model. We ran the models through the dataset and obtained the following results.")
    
    # Load image from URL
    url0 = "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Confusion%20Matrix%20Accuracy%20Guarantee/Accuracy%20Results.png"
    response = requests.get(url0)
    # Check if the request was successful
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        # Display the image using Streamlit
        st.image(image, caption="This is the confusion matrix of our in house trained model.", use_column_width=True)
    else:
        st.error(f"Failed to load image. Status code: {response.status_code}")  # Display an error message
    
    st.subheader("Identifying the diseases")
    
    st.write("From the accuracy results, we can see that the resnet50, followed by the resnet34 and finally the resnet18 models were able to identify the retinal fundus images most accurately. However, we also learnt by research that by combining models together, we would be able to generate more accurate predictions. Hence, we experimeted with two different ensemble model architectures, one by combining the resnet34 and the resnet18 models only, and the other by combining the abovementioned two models along with our custom resnet18 model. As seen from the accuracy diagram above, these two ensemble models performed exceptionally well, with the former outperforming the resnet18 model and the latter outperforming the resnet34 model in all diseases. This latter model, named ResNet-AR112, is used along with resnet50 and resnet34 in our prediction model framework below.")

    st.subheader("Data Analysis")
    st.write("The following are the confusion matrix diagrams attached for each of the models utilised in the prediction framework.")
    # Load image from URL
    url1 = "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Confusion%20Matrix%20Accuracy%20Guarantee/ResNet_AR112%20Metric.png"
    response = requests.get(url1)
    # Check if the request was successful
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        # Display the image using Streamlit
        st.image(image, caption="This is the confusion matrix of our in house trained model.", use_column_width=True)
    else:
        st.error(f"Failed to load image. Status code: {response.status_code}")  # Display an error message
    
    # Load image from URL
    url2 = "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Confusion%20Matrix%20Accuracy%20Guarantee/resnet34%20metric.png"
    response = requests.get(url2)
    # Check if the request was successful
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        # Display the image using Streamlit
        st.image(image, caption="This is the confusion matrix of the PyTorch pretrained resnet34 model.", use_column_width=True)
    else:
        st.error(f"Failed to load image. Status code: {response.status_code}")  # Display an error message

    # Load image from URL
    url3 = "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Confusion%20Matrix%20Accuracy%20Guarantee/resnet50%20metric.png"
    response = requests.get(url3)
    # Check if the request was successful
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        # Display the image using Streamlit
        st.image(image, caption="This is the confusion matrix of the PyTorch pretrained resnet50 model.", use_column_width=True)
    else:
        st.error(f"Failed to load image. Status code: {response.status_code}")  # Display an error message

    st.write("As seen, all the models achieved comparative high levels of accuracy at diagnosing each of the diseases. However, both the resnet50 and the resnet34 were relatively weaker at accurately diagnosing images of diabetic retinopathy. We believe that this is the case due to the wider range of symptoms for diabetic retinopathy, causing it to have a lack of significant tell tale signs for the model to identify. However, the ensemble model performed better in this aspect, which we believe could be due to the fact-checking mechanisms of combining the outputs of the three models together with appropriate weighting.")
    
def zero_shot_model_page():
    st.title("0 - shot Model Page")
    st.write("This is the contact page.")
    
def chain_of_though_few_shot_model_page():
    st.title("Chain of thought few shot Model Page")
    st.write("This is the contact page.")
    
def codebooks_page():
    st.title("Codebooks")
    st.write("This is the contact page.")
    
def acknowledgements_and_photos_page():
    st.title("Last words")
    st.write("This is the contact page.")

# Create a dictionary of subpages
subpages = {
    "Home": home_page,
    "Setting the project": dataset_and_confusion_matrix_page,
    "Classical Models": classical_models_page,
    "0 - shot Model": zero_shot_model_page,
    "Chain of thought few shot Model": chain_of_though_few_shot_model_page,
    "Acknowledgements": acknowledgements_and_photos_page,
    "Codebooks": codebooks_page
}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(subpages.keys()))

# Display the selected page
subpages[page]()

