import streamlit as st
import requests

st.title("Abstract")

# Text input box
user_input = st.text_area("Implement your", height=200)

# When the user clicks the button
if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Send request to local Ollama model
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "gemma3:1b",
                        "prompt": user_input,
                        "stream": False
                    }
                )

                # Show the result
                if response.status_code == 200:
                    result = response.json()
                    st.success(result.get("response", "No response received."))
                else:
                    st.error(f"Server error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
