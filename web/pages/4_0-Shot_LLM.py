import streamlit as st
st.title("Zero-shot Model Page")
st.write("We proceeded to try to use Large Language Models (LLMs) to analyse our retinal fundus images.")
st.subheader("About LLMs")
st.write("LLMs are a form of Generative Artificial Intelligence (GenAI) model that comprehends and generates language text that can be comprehended by a human. They are used in almost if not all GenAI Chatbots that dominate school life, including ChatGPT by OpenAI and Google's Gemini AI.")
st.write("Recent advances in LLM technology has allowed some LLMs to have vision capabilities, i.e., they are able to receive image inputs, interpret these images and output understandable text about the image.")
st.subheader("About Zero-shot learning")
st.write("Zero-shot learning refers to the model being given images not shown during training and classifying them. In this case, we used the LLM model ChatGPT-4o, a model with vision capabilities, and gave it a small testing dataset of retinal fundus images to gauge its proficiency in diagnosing the diseases.")
