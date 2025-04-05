"""
Streamlit frontend for the Security Policy Chat Assistant.
"""
import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Configuration
API_URL = "http://127.0.0.1:5000/api/query"
API_CONFIG_URL = "http://127.0.0.1:5000/api/config"

# Page configuration
st.set_page_config(
    page_title="Security Policy Chat Assistant",
    page_icon="🔒",
    layout="centered"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Security Policy Assistant. I can help you understand NovaTech's security policies. What would you like to know?"}
    ]

if "use_custom_api_key" not in st.session_state:
    st.session_state.use_custom_api_key = False
    
if "custom_api_key" not in st.session_state:
    st.session_state.custom_api_key = ""
    
if "api_key_confirmed" not in st.session_state:
    st.session_state.api_key_confirmed = False

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.8rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        border-bottom-right-radius: 0.2rem;
        margin-left: 2rem;
    }
    .chat-message.assistant {
        background-color: #475063;
        border-bottom-left-radius: 0.2rem;
        margin-right: 2rem;
    }
    .chat-timestamp {
        font-size: 0.8rem;
        color: #abb2bf;
        margin-top: 0.5rem;
        align-self: flex-end;
    }
    .css-1n76uvr {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: API Key Configuration
st.sidebar.title("Settings")

# Add API Key toggle and input in the sidebar
st.sidebar.subheader("API Key Settings")
use_custom_key = st.sidebar.checkbox("Use your own API key", value=st.session_state.use_custom_api_key)

# Update the session state for API key toggle
if use_custom_key != st.session_state.use_custom_api_key:
    st.session_state.use_custom_api_key = use_custom_key
    st.session_state.api_key_confirmed = False  # Reset confirmation status when toggling

# Show API key input field if checkbox is checked
if st.session_state.use_custom_api_key:
    custom_key = st.sidebar.text_input(
        "Enter your OpenAI API key",
        type="password",
        value=st.session_state.custom_api_key,
        help="Your API key will be sent securely to the backend when you confirm."
    )
    
    # Update the session state for custom key
    if custom_key != st.session_state.custom_api_key:
        st.session_state.custom_api_key = custom_key
        st.session_state.api_key_confirmed = False  # Reset when key changes
        
    # Add confirmation button
    if st.sidebar.button("Confirm API Key"):
        if st.session_state.custom_api_key:
            # Create a placeholder in the sidebar for status messages
            status_placeholder = st.sidebar.empty()
            status_placeholder.info("Configuring API key...")
            
            try:
                # Send the API key to the backend for configuration
                response = requests.post(
                    API_CONFIG_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"api_key": st.session_state.custom_api_key})
                )
                
                if response.status_code == 200:
                    st.session_state.api_key_confirmed = True
                    status_placeholder.success("✅ API key configured successfully!")
                else:
                    status_placeholder.error(f"❌ Failed to configure API key: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                status_placeholder.error(f"❌ Error connecting to backend: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please enter an API key first.")
    
    # Display confirmation status
    if st.session_state.api_key_confirmed:
        st.sidebar.info("Using your custom API key for all queries.")
    
    # Add a note about API key usage
    st.sidebar.markdown("""
    **Note**: 
    1. You must click "Confirm API Key" to apply changes
    2. Charges may apply to your OpenAI account
    3. The key is only stored in your current browser session
    """)

# Restore default API key button if using custom key
if st.session_state.use_custom_api_key and st.session_state.api_key_confirmed:
    if st.sidebar.button("Restore Default API Key"):
        try:
            # Tell the backend to restore default key
            response = requests.post(
                API_CONFIG_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"use_default": True})
            )
            
            if response.status_code == 200:
                st.session_state.api_key_confirmed = False
                st.session_state.custom_api_key = ""
                st.session_state.use_custom_api_key = False
                st.sidebar.success("✅ Restored default API key!")
                st.experimental_rerun()  # Refresh the UI
            else:
                st.sidebar.error("❌ Failed to restore default API key.")
        except Exception as e:
            st.sidebar.error(f"❌ Error connecting to backend: {str(e)}")

# About section
with st.sidebar.expander("About this app", expanded=False):
    st.markdown("""
    This Security Policy Chat Assistant helps you understand NovaTech Dynamics' security policies by answering your questions.
    
    It uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on NovaTech's security policy document.
    
    **Features:**
    - Answers questions about security policies
    - Cites relevant sections of the policy
    - Provides contextual information
    
    For IT support, contact helpdesk@novatechdynamics.com
    """)

# Main chat interface
st.title("Security Policy Chat Assistant")
st.caption("Ask questions about NovaTech Dynamics' security policies")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to add a message to the chat history
def add_message(role: str, content: str) -> None:
    """
    Add a message to the chat history.
    
    Args:
        role (str): Role of the message sender ('user' or 'assistant')
        content (str): Content of the message
    """
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": role, "content": content, "timestamp": timestamp})

# Chat input
user_query = st.chat_input("Ask about NovaTech's security policies...")

# Process the user's input
if user_query:
    # Add user message to chat history
    add_message("user", user_query)
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show a "thinking" effect
        message_placeholder.markdown("Thinking...")
        
        try:
            # Prepare the request data
            request_data = {"query": user_query}
            
            # Send the query to the Flask backend
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(request_data),
                timeout=30
            )
            
            # Process response
            if response.status_code == 200:
                assistant_response = response.json()["response"]
                
                # Simulate typing effect
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.02)  # Faster typing speed
                    message_placeholder.markdown(full_response + "▌")
                
                # Show the complete response
                message_placeholder.markdown(assistant_response)
                
                # Add the assistant's response to history
                add_message("assistant", assistant_response)
            else:
                error_msg = f"Error: Received status code {response.status_code} from API.\n{response.text}"
                message_placeholder.markdown(error_msg)
                add_message("assistant", error_msg)
                
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Could not connect to the backend API. Please make sure the Flask server is running."
            message_placeholder.markdown(error_msg)
            add_message("assistant", error_msg)
        except requests.exceptions.Timeout:
            error_msg = "Error: The request timed out. The server might be processing a complex query."
            message_placeholder.markdown(error_msg)
            add_message("assistant", error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            message_placeholder.markdown(error_msg)
            add_message("assistant", error_msg)

# Add sample questions at the bottom
st.markdown("---")
st.subheader("Sample questions you can ask:")
cols = st.columns(2)

with cols[0]:
    st.markdown("""
    - What is NovaTech's password policy?
    - How is confidential data protected?
    - What is the data classification system?
    - Who can I contact with security questions?
    """)
    
with cols[1]:
    st.markdown("""
    - What happens during employee termination?
    - How often are security assessments conducted?
    - What is the principle of minimal privileges?
    - What security monitoring is in place?
    """)