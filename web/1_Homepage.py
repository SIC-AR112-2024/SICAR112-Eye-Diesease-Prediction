import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
#home page
#st.set_up_config(page_icon='web/more_images/Gap_Sem_Logo.png')


image = Image.open('web/more_images/Logo_Image.png')
resized_image = image.resize((int(image.width/3.5), int(image.height/3.5)))

    
# Display the resized image
st.image(resized_image, use_column_width=False)
#st.image('web/more_images/Logo_Image.png')
st.title("Analysing and Building Risk Prediction Frameworks on Retinal Fundus Images")
st.write("Welcome to the home page of our SIC Project - AR112!")

st.subheader("Introduction")
st.write("The goal of this project is to test the effectiveness of Large Language Models (LLMs) at identifiying eye diseases from retinal fundus images in comparison to traditional machine learning (ML) models like ResNets.")

st.subheader("Motivations")
st.write("We were inspired to embark on this project in order to aid both doctors and patients in the process for diagnosing eye diseases. Our app hopes to function as a quick screening tool to give patients a quick prediction as to whether they have an eye diseases, and inform their decision on whether to seek further medical attention. For doctors, we hope that our app serves as a second opinion to inform them during their decision making processes, and allows them to devote more time to handling complex cases and treating pataients with eye diseases.")