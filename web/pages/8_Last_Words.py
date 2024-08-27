import streamlit as st
st.title("Last words")

st.subheader("Sources / Software")
st.write("We utilised a dataset we found on Kaggle, while our coding was carried out on Google Colab and VSCode. Our assets and code files are stored in Github, and we used Git and Git LFS to store our model .pth files in our Github.")

st.subheader("Acknowledgements")
st.write("We would like to thank our mentor, Mr Tan Pang Jin, for the guidance and help he has provided to us throughout this project, and our SIC supervisor Ms Jorinda Au, for guiding us in our learning.")

st.subheader("Challenges")
st.markdown("""
- **Resource Limitations:** We had faced issues with acquiring resources to run our models and code, most notably with Graphic Processing Units (GPUs) that we used to train our model, and Data Storage on Git Large File Storage (lfs), which was required for the streamlit app to run our models. With limited resources, we learnt to be prudent and not consume resources unnecessarily to save costs and time. We also developed workarounds to get past resource limits, particularly with the langchain API call limit, where we resorted to native OpenAI syntax calling.
- **Library failure:** Some of our software libraries, most notably PIL (or Pillow) failed. These unexpected errors necessitated us to develop work
- **Software learning curve:** 
- **Unclear documentation:** 
- **Dataset Limitations:** 
- **Exploration out of the curriculum:** 
""")

st.subheader("Photo Repo")
st.image('web/more_images/Discussion.jpeg')
st.image('web/more_images/Thonk.jpeg')
st.video('web/more_images/Work.mp4')

