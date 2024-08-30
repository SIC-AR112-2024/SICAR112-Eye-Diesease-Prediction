import streamlit as st

# Embed a video from a URL (e.g., YouTube)
video_id = "1s5AbPg4iZJh8GhAEPmy1Fa7VGVeWSVWn"


# Construct the iframe code
iframe_code = f"""
<iframe src="https://drive.google.com/file/d/{video_id}/preview" width="640" height="480"></iframe>
"""
st.header('Video Walkthrough')
# Embed the video
st.write('Watch our website walkthrough video guide narrated by Artificial Intelligence 😊.')
# Embed the iframe in Streamlit
st.markdown(iframe_code, unsafe_allow_html=True)


