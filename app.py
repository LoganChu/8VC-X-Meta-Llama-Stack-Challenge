import streamlit as st
import requests
import json
from typing import Dict, Any

# Configure the page
st.set_page_config(
    page_title="LAW - Research Abstract Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Grok-like design
st.markdown("""
<style>
    /* Global styles */
    [data-testid="stAppViewContainer"] {
        background: #0a0a0a;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        max-width: 1200px;
        padding: 3rem 2rem;
    }
    
    /* Logo and title styling */
    .logo-container {
        position: relative;
        text-align: center;
        margin-bottom: 4rem;
        padding-top: 2rem;
    }
    
    .logo {
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(255,255,255,0.5));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        display: inline-block;
    }
    
    .logo::before {
        content: '';
        position: absolute;
        inset: -20px -40px;
        background: radial-gradient(circle at center, rgba(255,255,255,0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .logo:hover::before {
        opacity: 1;
    }
    
    /* Input field styling */
    .input-container {
        position: relative;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 999px;
        padding: 1.5rem 2rem;
        color: white;
        font-size: 1.1rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(255,255,255,0.3);
        box-shadow: 0 0 20px rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.08);
    }
    
    /* Button styling */
    .stButton > button {
        background: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 999px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s ease;
        width: auto;
        margin: 0 auto;
        display: block;
    }
    
    .stButton > button:hover {
        background: rgba(255,255,255,0.15);
        border-color: rgba(255,255,255,0.3);
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Message styling */
    .user-message, .assistant-message {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }
    
    .user-message:hover, .assistant-message:hover {
        background: rgba(255,255,255,0.08);
        transform: translateY(-2px);
    }
    
    /* Background gradient */
    .background-gradient {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at top right, rgba(255,255,255,0.1), transparent 60%),
                    radial-gradient(circle at bottom left, rgba(255,208,134,0.1), transparent 60%);
        pointer-events: none;
        z-index: -1;
    }
</style>

<!-- Background gradient -->
<div class="background-gradient"></div>

<!-- Logo and title -->
<div class="logo-container">
    <h1 class="logo">LAW</h1>
</div>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Main input area
st.markdown('<div class="input-container">', unsafe_allow_html=True)
topic = st.text_input("", 
                     placeholder="Enter your research title...",
                     help="Enter a clear and concise title for your research paper")

if topic:
    context = st.text_area("", 
                         placeholder="Enter keywords and context for your research...",
                         height=100)
    
    if st.button("Generate Abstract"):
        with st.spinner("✍️ Generating abstract..."):
            try:
                payload = {
                    "topic": topic,
                    "context": context,
                    "temperature": st.session_state.get('temperature', 0.7),
                    "max_length": st.session_state.get('max_length', 500)
                }
                
                response = requests.post(
                    "http://localhost:8000/generate_paper",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Title: {topic}\nContext: {context}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get("abstract", "No content generated.")
                    })
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Display chat history
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            content = message["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="user-message"><strong>Input:</strong><br>{content}</div>',
                unsafe_allow_html=True
            )
        else:
            content = message["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="assistant-message"><strong>Generated Abstract:</strong><br>{content}</div>',
                unsafe_allow_html=True
            )

# Sidebar for settings
with st.sidebar:
    st.markdown('<div class="settings-header">Generation Settings</div>', unsafe_allow_html=True)
    temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                          help="Higher values produce more creative but potentially less focused abstracts")
    max_length = st.slider("Length", min_value=100, max_value=1000, value=500, step=50,
                          help="Maximum length of the generated abstract") 