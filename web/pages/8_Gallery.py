import streamlit as st
st.title("Last words")

st.subheader("Sources / Software")
st.write("We utilised a dataset we found on Kaggle, while our coding was carried out on Google Colab and VSCode. Our assets and code files are stored in Github, and we used Git and Git LFS to store our model .pth files in our Github.")

st.subheader("Acknowledgements")
st.write("We would like to thank our mentor, Mr Tan Pang Jin, for the guidance and help he has provided to us throughout this project, and our SIC supervisor Ms Jorinda Au, for guiding us in our learning.")

st.subheader("Challenges")
st.markdown("""
- **Resource Limitations:** We had faced issues with acquiring resources to run our models and code, most notably with Graphic Processing Units (GPUs) that we used to train our model, and Data Storage on Git Large File Storage (lfs), which was required for the streamlit app to run our models. With limited resources, we learnt to be prudent and not consume resources unnecessarily to save costs and time. We also developed workarounds to get past resource limits, particularly with the langchain API call limit, where we resorted to native OpenAI syntax calling.
- **Library failure:** Some of our software libraries, most notably PIL (or Pillow) failed. These unexpected errors necessitated us to develop workarounds and utilise alternative libraries so as to maintain the functionality of our code.
- **Software learning curve:** The software that we used was complex in nature, which made it challenging for us to learn in the short timeframe we had. We resorted to documentation, forums like stack-overflow and even generative AI chatbots (e.g. Gemini, ChatGPT) to brush up on our knowledge. After putting in much time and effort even after our working hours, we were able to master these softwares to a sufficient degree to attempt using them. These included Amazon Web Service Cloud Storage, git-annex and the aforementioned langchain.
- **Unclear documentation:** The current documentation for machine learning models has not been advanced yet, hence most of the documentation were unable to solve the issues that we were facing in designing our models and our website. This required us to spend more time understanding what each indivdiual line of code meant, which inadvertently enriched our learning experience. 
- **Dataset Limitations:** We were also constrained by the quality of images of our dataset, and we could improve on this in the future as the quality of data increases.
- **Exploration out of the curriculum:** Our project was highly interdisciplinary in nature, spanning across computing, mathematics, linguistics and biology. We were able to capitalise on each others strengths to cover up for our own weaknesses to gain a deeper and fuller understanding about the workings of our project and build on our understanding.
""")

st.subheader("Further Research")
st.markdown("""
- **Other Applications:** With the foundational script for training images already set up, we can potentially branch out to other fields which require computer vision capabilities and test our system to see if it is still applicable and effective in that use case.
- **More Refined Dataset:** We could expand our dataset to include images of other diseases, as well to impose quality checks to make sure that only images of high quality are being scanned by our model.
- **Retrieval Augmented Generation (RAG):** We could have added medical texts that explain how to diagnose these diseases so as to improve the accuracy of the LLM model in diagnosising diseases.
- **Explore more prompting frameworks:** We could have explore more prompting frameworks, like the react framework.
- **Improve explainability:** We could have set up an evaluation metric for explainability, allowing us to improve the explainability of the results by the LLM to produce more accurate findings.
- **Improve on prompting frameworks:** We could add more prompts to the Chain of Thought (CoT) Few Shot prompting mechanism in future in order for the LLM to have more reference points in order to make a more accurate prediction.
""")

st.subheader("Photo Repo")

st.write('The environment')
st.image('web/more_images/SMU_School_of_Economics.jpeg', caption="The building next to us")
st.video('web/more_images/Road.mp4')
st.image('web/more_images/Subway1.jpeg', caption="Our reporting venue")
st.image('web/more_images/Subway2.jpeg')

st.write("Research")
st.image('web/more_images/Video1.jpeg', caption="Watching YouTube videos to learn more")
st.image('web/more_images/Video2.jpeg')
st.image('web/more_images/Video3.jpeg')
st.image('web/more_images/Discussion.jpeg', caption="Discussion")
st.image('web/more_images/Whiteboard_Notes.jpeg', caption="Notes")
st.image('web/more_images/Diagram.jpeg')
st.image('web/more_images/Pinecone.jpeg')
st.image('web/more_images/Research.jpeg')

st.write("Coding process")
st.image('web/more_images/Begin_Coding.jpeg', caption="The journey begins")
st.image('web/more_images/Thonk.jpeg', caption="How did we take this picture?")
st.video('web/more_images/Work.mp4')
st.image('web/more_images/CY_not_anger.jpeg')
st.image('web/more_images/CY_anger.jpeg')
st.image('web/more_images/SBS1.jpeg')
st.image('web/more_images/SBS2.jpeg')
st.image('web/more_images/SBS_anger.jpeg')
st.video('web/more_images/SBS.mp4')

st.write("Running our models")

st.image('web/more_images/!pip_install.jpeg')
st.image('web/more_images/Error.jpeg', caption="One of many errors")
st.video('web/more_images/Error.mp4')
st.image('web/more_images/Epoch.jpeg', caption='Training...')
st.video('web/more_images/Time_Lapse1.mp4')
st.video('web/more_images/Time_Lapse2.mp4')
st.image('web/more_images/Improvement.jpeg', caption="The machine learns")
