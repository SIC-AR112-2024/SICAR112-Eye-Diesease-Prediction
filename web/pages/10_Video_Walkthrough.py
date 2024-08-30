import streamlit as st

# Embed a video from a URL (e.g., YouTube)
video_url = "https://drive.google.com/file/d/1s5AbPg4iZJh8GhAEPmy1Fa7VGVeWSVWn/view?usp=sharing"
test_vid = "https://www.youtube.com/watch?v=iBuTEywEQ6U"

st.header('Video Walkthrough')
# Embed the video
st.write('Watch our website walkthrough video guide narrated by Artificial Intelligence 😊.')
st.video(test_vid)