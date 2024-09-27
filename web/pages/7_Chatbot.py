import streamlit as st
import openai
from time import sleep
# from backend.zero_shot import (init_prompt_zero_shot)
# from backend.few_shot import (init_prompt_CoT, query_few_shot)
import google.generativeai as genai
from tenacity import (retry, stop_after_attempt, wait_random_exponential)
import os

API_KEY = os.environ["OPENAI_API_KEY"]
if API_KEY is not None:
    print("Secret Key Retrieved Successfully")
    openai.api_key = API_KEY
else:
    print("Failed to retrieve the Secret Key")


images = {
    'Glaucoma':'https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/glaucoma/663.jpg?raw=true', #Glaucoma
    'Diabetic Retinopathy':'https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/diabetic_retinopathy/342.jpg?raw=true', #Diabetic Retinopathy
    'Cataract':'https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/cataract/110.jpg?raw=true'} #Cataract
message = []
def init_prompt_CoT():
    return [
    {'role': 'system',
        'content': """You are a medical student. You will be given several retinal fundus images as a test.
Firstly, describe key features depicted in the image, of no less than 100 words, such as the macula, optic nerve, optic cup and disc and retinal blood discs.
If the eye is healthy, say \"HEALTHY\". If not, tell me whether the patient has \"CATARACT\", \"DIABETIC RETINOPATHY\", or \"GLAUCOMA\". Your final diagnosis must be strictly 1 or 2 words, on a new line."""
    },
    {"role": "user",
        "content": [
            {"type": "text", "text": "Produce a diagnosis for the following:"},
            {"type": "image_url", "image_url": {
                "url": "https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/diabetic_retinopathy/342.jpg?raw=true",
            },
            },
        ],
    },
    {"role": "assistant",
    "content": """The fundus image shows numerous hard exudates scattered throughout the retina, particularly in the macula. These are yellowish-white deposits that are a hallmark of diabetic retinopathy. Additionally, there are microaneurysms, which are small, dilated blood vessels that leak fluid. These features, along with the presence of retinal hemorrhages, are consistent with diabetic retinopathy. The optic nerve appears healthy, ruling out glaucoma. There is no evidence of a cataract.
DIABETIC RETINOPATHY."""
    },
    {"role": "user",
        "content": [
            {"type": "text", "text": "Produce a diagnosis for the following:"},
            {"type": "image_url", "image_url": {
                "url": "https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/glaucoma/883.jpg?raw=true",
            },
            },
        ],
    },
    {"role": "assistant",
    "content": """The image shows a view of the fundus of the eye. The optic disc, the point where the optic nerve leaves the eye, is visible in the center of the image. The disc is slightly pale and the margins appear blurred. The macula, the central region responsible for central vision, appears slightly edematous. The retinal blood vessels are clearly visible and appear normal.
GLAUCOMA"""}]

st.title("LLM Playground")
st.markdown("""Below, we have the ability to query GPT-4o using 0-shot and few-shot CoT prompting.
For more information on how prompting helps LLMs, visit the corresponding pages in the sidebar.""")
ailment = st.selectbox("Pick a disease to diagnose:", ('Glaucoma', 'Diabetic Retinopathy', 'Cataract'))
LLM_mode = st.selectbox("Pick a prompting method:", ('0-shot', 'Few-shot'))
#API_Key = st.text_input("API Key here:", placeholder="Type API Key (Ask us for ours!)", type="password", value=OPENAI_API_KEY)

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
    if st.button('Generate GPT-4o output (0-shot):'):
        client = openai.OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            max_tokens=300,
            stream=True
        )
        with st.chat_message('ai'):
            st.write_stream(response)
if LLM_mode == 'Few-shot':
    if st.button('Generate GPT-4o output (0-shot):'):
        client = openai.OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            max_tokens=300,
            stream=True
        )
        with st.chat_message('ai'):
            st.write_stream(response)