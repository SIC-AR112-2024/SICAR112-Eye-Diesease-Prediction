import streamlit as st
import openai
from time import sleep
from backend.zero_shot import (init_prompt_zero_shot)
from backend.few_shot import (init_prompt_CoT, query_few_shot)
images = {
    'Glaucoma':'https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/diabetic_retinopathy/342.jpg?raw=true', #Glaucoma
    'Diabetic Retinopathy':'https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/diabetic_retinopathy/342.jpg?raw=true', #Diabetic Retinopathy
    'Cataract':'https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/diabetic_retinopathy/342.jpg?raw=true'} #Cataract
message = []

def hello_my_name_is_markiplier(text):
    for line in text.split('\n'):
        for word in text.split(' '):
            yield word + ' '
            sleep(0.05)
        yield '\n'


st.title("LLM Playground")
st.markdown("""Below, we have the ability to query GPT-4o using 0-shot and few-shot CoT prompting.
For more information on how prompting helps LLMs, visit the corresponding pages in the sidebar.""")
ailment = st.multiselect("Pick a disease to diagnose:", ['Glaucoma', 'Diabetic Retinopathy', 'Cataract'])
LLM_mode = st.multiselect("Pick a prompting method:", ['0-shot', 'Few-shot'])
API_Key = st.text_input("API Key here 👇", placeholder="Type API Key (Ask us for ours!)")

if LLM_mode == '0-shot':
    message = [
    {'role':'user',
    'content':[
        {'type': 'text', 'text':'''You are a medical student. You will be given several retinal fundus images as a test.
Firstly, describe key features depicted in the image, of no less than 100 words, such as the macula, optic nerve, optic cup and disc and retinal blood discs.
If the eye is healthy, say "HEALTHY". If not, tell me whether the patient has "CATARACT", "DIABETIC RETINOPATHY", or "GLAUCOMA". Your final diagnosis must be strictly 1 or 2 words, on a new line.'''},
        {'type': 'image_url', 'image_url':{'url':images[ailment]}}
    ]}]
if LLM_mode == 'Few-shot':
    message = []
    message = init_prompt_CoT()
    message.append({"role": "user",
            "content": [
                {"type": "text", "text": "Produce a diagnosis for the following:"},
                {"type": "image_url", "image_url": {
                    "url": images[ailment]
    }}]})

for i in message:
    if i['role'] == 'user':
        with st.chat_message('user'):
            st.write(i['content'][0]['text'])
            st.image(i['content'][1]['image_url']['url'])
    elif i['role'] == 'system':
        with st.chat_message('System Instructions', avatar = ':material/computer:'):
            st.write(i['content'])
    else:
        with st.chat_message('ai'):
            st.write(i['content'])

if LLM_mode == '0-shot':
    st.button('Generate GPT-4o output (0-shot):')
if LLM_mode == 'Few-shot':
    st.button('Generate GPT-4o output (few-shot):')