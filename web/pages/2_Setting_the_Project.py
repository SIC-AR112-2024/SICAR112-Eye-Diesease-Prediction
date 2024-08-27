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
        
        
st.subheader('Review of the current Literature')
st.write('Utilising Artificial Intelligence (AI) in identifying eye diseases from retinal imaging is a rapidly developing area in research. Machine learning models, particularly deep learning algorithms, have shown great success in detecting and grading diabetic retinopathy from fundus images, and some are able to achieve performance comparable to or even surpassing that of experienced ophthalmologists in some cases. Currently, researchers are working on improving the accuracy, efficiency, and interpretability of these models. Some common diseases being targeted are age-related macular degeneration, glaucoma and diabetic retinopathy, of which the latter two are also covered by our models. Scientists are also applying novel architectures like Convolutional Neural Networks (CNN), which was also tested in our project. Currently, some challenges include, improving model generalization across diverse populations and imaging equipment, developing models that can work with limited labeled data, and enhancing interpretability and explainability of AI decisions. In our project, we have adopted more powerful Residual Network (ResNet) models which would be able to generate more accurate predictions with limited data. Regarding explainability, we strongly believe that this is an area where Large Language Models (LLMs) can be used in a multimodal AI system due to their ability to output understandable human language.')

st.subheader('Relevant Papers')
st.markdown("[1. Applications of deep learning for detecting ophthalmic diseases with ultrawide-field fundus images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10754665/)")
st.markdown("[2. Automated machine learning model for fundus image classification by health-care professionals with no coding experience](https://www.nature.com/articles/s41598-024-60807-y)")
st.markdown("[3. Development of a Fundus Image-Based Deep Learning Diagnostic Tool for Various Retinal Diseases](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8142986/)")
st.markdown("[4. A comprehensive review of artificial intelligence models for screening major retinal diseases](https://link.springer.com/article/10.1007/s10462-024-10736-z)")
st.markdown("[5. Development and Validation of Deep Learning Models for Screening Multiple Abnormal Findings in Retinal Fundus Images](https://pubmed.ncbi.nlm.nih.gov/31281057/)")
st.markdown("[6. Effectiveness of artificial intelligence screening in preventing vision loss from diabetes: a policy model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10042864/)")
st.markdown("[7. Prevalence, Risk Factors, and Impact of Undiagnosed Visually Significant Cataract: The Singapore Epidemiology of Eye Diseases Study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5271362/#:~:text=The%20overall%20age%2Dstandardized%20prevalence,aged%2060%20years%20or%20older.)")
st.markdown("[Implementing and evaluating a fully functional AI-enabled model for chronic eye disease screening in a real clinical environment](https://bmcophthalmol.biomedcentral.com/articles/10.1186/s12886-024-03306-y)")