from dotenv import load_dotenv
import openai
import streamlit as st
import os
from PIL import Image
from google.cloud import vision
import io

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("API key not found. Please make sure to set OPENAI_API_KEY in your .env file.")

# Function to load OpenAI model and get responses
def get_openai_response(input, image):
    try:
        # First, we'll need to upload the image to OpenAI's servers for processing
        if image:
            # Convert the image to bytes so it can be uploaded to OpenAI
            client = vision.ImageAnnotatorClient()
            image_bytes = image_to_byte_array(image)
            vision_image = vision.Image(content=image_bytes)
            response = client.label_detection(image=vision_image)
            labels = response.label_annotations
            description = ", ".join([label.description for label in labels])
            prompt = input + f"\nThe image seems to contain: {description}" if input else f"The image contains: {description}"

        else:
            prompt = input  # If no image, just use the input text

        # Make a call to OpenAI API for image or text generation
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are an AI that helps users understand images."},
                    {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )


        return response.choices[0].message['content'].strip()


    except Exception as e:
        return f"Error: {str(e)}"

# Helper function to convert image to byte array
def image_to_byte_array(image: Image) -> bytes:
    from io import BytesIO
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

# Initialize our Streamlit app
st.set_page_config(
    page_title="OpenAI Image and Text Application",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add a header with custom styles
st.markdown("<h1>AI-Powered Image & Text Description Generator</h1>", unsafe_allow_html=True)


# Create columns for input and image
col1, col2 = st.columns([2, 1])

# Input column
with col1:
    st.subheader("Input Prompt")
    input = st.text_area("Enter your prompt here:", height=200, placeholder="Type your prompt...")

# Image column
with col2:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

# Submit button
submit = st.button("Tell me about the image", use_container_width=True)

# If submit button is clicked
if submit:
    if image is None:
        st.warning("Please upload an image first.", icon="⚠️")
    else:
        with st.spinner("Generating response..."):
            response = get_openai_response(input, image)
        st.success("Response generated!", icon="✅")
        st.write(response)

