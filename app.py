import openai
import os
from PIL import Image
import io
from dotenv import load_dotenv
import streamlit as st
from google.cloud import vision


load_dotenv()  # This loads .env file variables
openai.api_key = os.getenv("OPENAI_API_KEY")



# Explicitly set the path to the credentials using the env variable
GOOGLE_CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_CREDS_PATH:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env")

# Set the environment variable used by the Google Cloud client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS_PATH
print("Using credentials from:", GOOGLE_CREDS_PATH)  # Debug log

# Initialize the Google Vision API client
client = vision.ImageAnnotatorClient()

# Function to use Google Vision API for image recognition
def recognize_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    content = image_bytes.getvalue()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    if response.error.message:
        return f"Error: {response.error.message}"
    
    labels_desc = [label.description for label in labels]
    image_description = ", ".join(labels_desc)
    return image_description

# Function to generate description using GPT-4o (chat-based)
def get_gpt4o_description(input_text, user_message):
    if not input_text:
        return "I couldn't detect any meaningful labels from the image."

    messages = [
        {
            "role": "system",
            "content": "You are an AI that describes image contents in detail based on detected labels and user context."
        },
        {
            "role": "user",
            "content": f"The image appears to contain: {input_text}. Based on this, respond to the user's message: '{user_message}'."
        }
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating description: {str(e)}"


# Function to handle text-based responses
def get_gpt4o_text_response(input_text):
    messages = [
        {"role": "system", "content": "You are an assistant that provides helpful and informative responses."},
        {"role": "user", "content": input_text}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit setup
st.set_page_config(
    page_title="AI Image & Text Description Generator",
    page_icon=":camera:",
    layout="wide"
)

st.title("AI-Powered Image & Text Description Generator")

# Text prompt
text_prompt = st.text_area("Enter your text prompt", height=200, placeholder="Type your prompt...")

if st.button("Generate Response for Text"):
    if text_prompt.strip() != "":
        with st.spinner("Generating response..."):
            text_response = get_gpt4o_text_response(text_prompt)
        st.success("Response generated successfully!")
        st.write(text_response)
    else:
        st.warning("Please enter a text prompt.", icon="⚠️")

# Image uploader
uploaded_image = st.file_uploader("Upload an image for description", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Generate Description for Image"):
        with st.spinner("Analyzing image..."):
            image_description = recognize_image(image)
            print("Google Vision Labels:", image_description)


            if "Error" in image_description:
                st.error(image_description)
            else:
                if text_prompt.strip() != "":
                    detailed_description = get_gpt4o_description(image_description, text_prompt)
                    st.success("Description generated successfully!")
                    st.write(detailed_description)
                else:
                    detailed_description = get_gpt4o_description(image_description, "Please analyze the image and describe what you see.")
                    st.success("Description generated successfully!")
                    st.write(detailed_description)
