import streamlit as st
from PIL import Image
import io

st.title('Image Store')
st.write('As part of the GAPSem Congress 2024, we have come up with a mini activity! We have taken some images from our database and put them here for you to download them and run through our Applet. Help each patient receive appropriate treatment by accurately diagnosing each image. Good luck ;).')

imageURLs = ['dataset/cataract/4.jpg', 'dataset/normal/6.jpg', 'dataset/diabetic_retinopathy/17.jpg', 'dataset/diabetic_retinopathy/49.jpg', 'dataset/glaucoma/9.jpg', 'dataset/cataract/110.jpg', 'dataset/normal/12.jpg', 'dataset/glaucoma/46.jpg']
names = ['Adam', 'Bob', 'Carlos', 'Donnie', 'Edgar', 'Freddy', 'Gupta', 'Hafiz']
images = [None] * 8
img_byte_arrs = [None] * 8


for i in range(len(imageURLs)):
    # Load or create an image
    images[i] = Image.open(imageURLs[i])  # Replace with your image file path or create an image dynamically

    if images[i].mode == 'RGBA':
        images[i] = images[i].convert('RGB')  # Convert to RGB (no alpha)

    # Convert the image to bytes for download
    img_byte_arrs[i] = io.BytesIO()
    images[i].save(img_byte_arrs[i], format='JPEG')  # You can change format if needed (PNG, etc.)
    img_byte_arrs[i] = img_byte_arrs[i].getvalue()

    # Display the image in Streamlit
    st.image(images[i], caption="Patient " + str(i) + ": " + names[i], use_column_width=True)

    # Provide a download button for the image
    st.download_button(
        label="Download Retinal Fundus Image",
        data=img_byte_arrs[i],
        file_name= names[i]+ ".jpg",  # The name of the file to be downloaded
        mime="image/jpeg"  # Adjust mime type depending on the image format
    )