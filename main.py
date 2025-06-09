import streamlit as st
import requests
import os
from dotenv import load_dotenv

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Multi-Model Chatbot", layout="centered")

# --- API Key Loading ---
# Load API key from Streamlit secrets or .env
API_KEY = None
try:
    API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    # If not in secrets, try loading from .env file
    load_dotenv()
    API_KEY = os.getenv("OPENROUTER_API_KEY")

# Validate API key
if not API_KEY:
    st.error("🚨 Missing API Key. Please set OPENROUTER_API_KEY in your .env file or Streamlit secrets.")
    st.stop() # Stop the app if no API key is found



# --- Supported Models ---
MODELS = {
    "LLaMA 3 (8B)": "meta-llama/llama-3-8b-instruct",
    "DeepSeek R1": "deepseek/deepseek-r1",
    "DeepSeek Chat V3": "deepseek/deepseek-chat-v3-0324",
    "Gemma 3 27B": "google/gemma-3-27b-it",
    "Mistral Small": "mistralai/devstral-small"
}

# --- Function to query OpenRouter API ---
@st.cache_data(show_spinner=False) # Cache API responses to prevent re-fetching same data on reruns
def query_openrouter(model_id: str, messages: list) -> str:
    """
    Queries the OpenRouter API with the specified model and message history.

    Args:
        model_id (str): The ID of the model to use (e.g., "meta-llama/llama-3-8b-instruct").
        messages (list): A list of message dictionaries in the format
                         [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}].

    Returns:
        str: The content of the assistant's reply or an error message.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Multi-Model Chatbot" # Custom title for OpenRouter logs
    }
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7 # Adjust temperature for creativity/determinism
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {response.status_code} {response.reason} for url: {url}\nDetails: {response.text}"
    except KeyError:
        return "❌ Error: Could not parse API response or unexpected format."

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model_name = st.selectbox("Choose a model:", list(MODELS.keys()))
    selected_model_id = MODELS[selected_model_name]
    
    # Using st.toggle for a more modern switch appearance
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    
    st.markdown("---")
    st.caption("Powered by [OpenRouter](https://openrouter.ai)")

# --- Theme Configuration (applying custom CSS once) ---
# This CSS primarily affects the overall page background and text color.
# Streamlit's st.chat_message handles individual message bubble colors.
bg_color = "#1e1e1e" if dark_mode else "#ffffff"
text_color = "white" if dark_mode else "black"

st.markdown(f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stChatMessage {{ /* This targets Streamlit's default chat message container */
            color: {text_color}; 
        }}
        /* Further custom styles for specific elements if needed, though st.chat_message is well-styled */
    </style>
""", unsafe_allow_html=True)


# --- Main Chat Interface ---
st.markdown(f"<h2 style='text-align: center; color:{text_color}'>🤖 Multi-Model Chatbot</h2>", unsafe_allow_html=True)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages using st.chat_message
for msg in st.session_state.chat_history:
    # st.chat_message automatically styles messages based on the role ('user' or 'assistant')
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input prompt
user_prompt = st.chat_input("Type your message here...")

# Process user input
if user_prompt:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get assistant's reply
    with st.spinner(f"Thinking with {selected_model_name}..."):
        # Pass the full chat history to maintain conversation context
        reply = query_openrouter(selected_model_id, st.session_state.chat_history)
    
    # Add assistant's reply to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    
    # Display the assistant's message
    with st.chat_message("assistant"):
        st.markdown(reply)