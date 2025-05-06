from dotenv import load_dotenv
import openai
import streamlit as st
import os

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("API key not found. Please make sure to set OPENAI_API_KEY in your .env file.")

# Function to get a response from OpenAI's API
def get_openai_response(question):
    try:
        # Make a call to the OpenAI API (you can choose different models, e.g., 'text-davinci-003', 'gpt-4', etc.)
        response = openai.Completion.create(
            model="text-davinci-003",  # You can switch to 'gpt-4' or 'gpt-3.5-turbo' based on your use case
            prompt=question,
            max_tokens=150  # You can adjust the token limit as needed
        )
        return response.choices[0].text.strip()  # Return the response text
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")

st.header("OpenAI Q&A Application")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input field for the user query
input = st.text_input("Input: ", key="input")

submit = st.button("Ask the question")

# If the ask button is clicked, process the input
if submit and input:
    response = get_openai_response(input)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", input))
    st.subheader("The Response is")
    st.write(response)
    st.session_state['chat_history'].append(("Bot", response))

# Display chat history
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
