import streamlit as st
import requests
from PIL import Image
import requests
from io import BytesIO

st.title("Classical Models")
st.subheader("About ResNet")
st.write("ResNet, which stands for residual neural network, is deep learning architecture which involves the use of “residual connections”, where the input from a few blocks ago is directly added to the current block’s output. This helps to prevent the vanishing gradient problem, where the updates to the model during training become very small as the number of layers increase. This allows us to build larger networks that can infer more features from the image.")
st.subheader("About GoogLeNet")
st.write("GoogLeNet is a Convolutional Neural Network that makes use of Inception modules, where multiple filters of different sizes are applied at once to extract different levels of features (small, finer to large, overall). This improves the accuracy of the model, especially for our purposes where both small (e.g. presence of red spots) and larger (e.g. blood vessels) details are essential to diagnosing the images.")
st.subheader("About Ensemble Model")
st.write("The ensemble model feeds the input image into multiple models and takes a weighted average of their outputs to obtain an overall diagnosis. This improves accuracy as now multiple models must produce a wrong diagnosis to impact the overall diagnosis.")
st.subheader("Use of GPUs")
st.write("For this project, we needed to utilise Graphic Processing Units (GPUs) with sufficient Random Access Memory (RAM) in order to run the model. This imposed a strict computational limitation on our models, most notably on our highly computationally expensive GoogLeNet model, which we had to streamline and simplify drastically in order for our computer to have sufficient GPU RAM to run it. All of our models were trained on an L4 GPU. However, as this GPU costs money, all models except the ensemble models, the ResNet50 and the GoogLeNet Model can be run on the free T4 GPU. This resource limitation limited the number of layers that we could implement in our custom models, which could have affected the efficacy of the custom ResNet18 and GoogLeNet Model. This also affected our decision on what models to put into our ensemble model because the computationally demanding ResNet50 model could not be combined with other models in the ensemble model architecture without taking up too much GPU RAM. However, for evaluation, these expensive computational resources are not required as the model evaluation loop can run sufficiently quickly on the CPU of the computer.")
st.subheader("Results")
st.write("Firstly, we tried to analyse the retinal fundus images using a few models. These include the PyTorch pretrained ResNet18, ResNet34 and ResNet50 models, our own custom ResNet18 model, a Convolutional Neural Network model that we built, a GoogLeNet model, and an ensemble model comprising of a combination of the previously mentioned model. We ran the models through the dataset and obtained the following results.")
# Load image from URL
url0 = "https://raw.githubusercontent.com/SIC-AR112-2024/SICAR112-Eye-Disease-Prediction/main/Confusion%20Matrix%20Accuracy%20Guarantee/Accuracy%20Results.png"
response = requests.get(url0)
# Check if the request was successful
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    # Display the image using Streamlit
    st.image(image, caption="Table of the accuracy results of our various models.", use_column_width=True)
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
st.subheader("Comparisons with state-of-the-art models")
st.write("")
