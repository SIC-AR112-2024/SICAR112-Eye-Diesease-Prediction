import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
#home page
#st.set_up_config(page_icon='web/more_images/Gap_Sem_Logo.png')

def get_window_width():
    # Custom HTML/JS to get window width
    component_code = """
    <script>
    window.onload = function() {
        const width = window.innerWidth;
        window.parent.postMessage({ type: 'windowWidth', width: width }, '*');
    };
    </script>
    """
    # Create an empty component to run the script
    components.html(component_code, height=1)
    
    # Receive the window width from JavaScript
    with st.expander("Hidden"):
        width = st.session_state.get("windowWidth", 800)  # Default width if not set
    return width

window_width = get_window_width()
image = Image.open('web/more_images/Logo_Image.png')
resized_image = image.resize((int(image.width/2), int(image.height/2)))

    
# Display the resized image
st.image(resized_image, use_column_width=False)
#st.image('web/more_images/Logo_Image.png')
st.title("SIC AR112 2024 Landing Page")
st.write("Welcome to the home page of our SIC Project - AR112!")

st.subheader("Introduction")
st.write("The goal of this project is to test the effectiveness of Large Language Models (LLMs) at identifiying eye diseases from retinal fundus images in comparison to traditional machine learning (ML) models like ResNets.")

st.subheader("Motivations")
st.write("We were inspired to embark on this project in order to aid both doctors and patients in the process for diagnosing eye diseases. Our app hopes to function as a quick screening tool to give patients a quick prediction as to whether they have an eye diseases, and inform their decision on whether to seek further medical attention. For doctors, we hope that our app serves as a second opinion to inform them during their decision making processes, and allows them to devote more time to handling complex cases and treating pataients with eye diseases.")