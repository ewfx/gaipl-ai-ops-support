import streamlit as st
import requests
from database_conn import DB
from datasets import load_dataset

# Hugging Face API details
HF_API_TOKEN = "YOUR_SECRET"
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Load ServiceNow Insight Bench dataset
#dataset = load_dataset("ServiceNow/insight_bench")
dataset = load_dataset("arsen-r-a/incident-management-qa-test1")

def get_relevant_data(user_input):
    """Finds the most relevant dataset entry based on user input."""
    for example in dataset["train"]:
        if user_input.lower() in example["question"].lower():  # Match question
            return example["answer"]  # Return answer if found
    return "No relevant information found in the dataset."

def call_huggingface_api(prompt, context=""):
    """Calls the Hugging Face Inference API with dataset context."""
    full_prompt = f"Context: {context}\n\nUser: {prompt}\nBot:"
    
    payload = {"inputs": full_prompt, "parameters": {"max_length": 200, "num_return_sequences": 1}}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"API Error: {response.json()}"

#################
# Streamlit UI
st.title("🤖 Chatbot Support (Powered by insight_bench)")

st.write("✅ Streamlit is running...")

# Check if session is new
if "session_started" not in st.session_state:
    st.session_state.session_started = True
    DB.clear_chat_history()  # Clears chat history at the start of a new session

# Retrieve chat history from database
st.subheader("Chat History")
chat_history = DB.get_chat_history()
for user_msg, bot_msg, timestamp in chat_history:
    st.write(f"🧑 {user_msg} (at {timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
    st.write(f"🤖 {bot_msg}")
    st.write("---")

# Input text box
user_input = st.text_input("You:", "")

# If user enters text, generate a response
if user_input:
    try:
        context_data = get_relevant_data(user_input)  # Fetch relevant dataset info
        bot_response = call_huggingface_api(user_input, context_data)
        DB.save_chat_to_db(user_input, bot_response)
        st.write("🤖 Chatbot:", bot_response)
    except Exception as e:
        st.error(f"Error occurred: {e}")
