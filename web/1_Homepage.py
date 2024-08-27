import streamlit as st
#home page
#st.set_up_config(page_icon='web/more_images/Gap_Sem_Logo.png')
st.image('web/more_images/Gap_Sem_Logo.png')
st.title("SIC AR112 2024 Landing Page")
st.write("Welcome to the home page of our SIC Project - AR112!")

st.subheader("Introduction")
st.write("The goal of this project is to test the effectiveness of Large Language Models (LLMs) at identifiying eye diseases from retinal fundus images in comparison to traditional machine learning (ML) models like ResNets.")

st.subheader("Motivations")
st.write("We were inspired to embark on this project in order to aid both doctors and patients in the process for diagnosing eye diseases. Our app hopes to function as a quick screening tool to give patients a quick prediction as to whether they have an eye diseases, and inform their decision on whether to seek further medical attention. For doctors, we hope that our app serves as a second opinion to inform them during their decision making processes, and allows them to devote more time to handling complex cases and treating pataients with eye diseases.")