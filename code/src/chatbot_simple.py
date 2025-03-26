import streamlit as st
from transformers import pipeline
from database_conn import DB
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
import torch
import pandas as pd
from huggingface_hub import login

# Replace 'your_huggingface_token' with your actual token
login(token="YOUR_SECRET")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ServiceNow Insight Bench dataset
servicenow_dataset = load_dataset("json", data_files="insight_bench.json")

# Load Incident Management QA dataset
#incident_mgmt_dataset = load_dataset("arsen-r-a/incident-management-qa-test1")

# Print sample data
print(servicenow_dataset["train"][0])  # First example from ServiceNow dataset
#print(incident_mgmt_dataset["train"][0])  # First example from Incident Management dataset

# Define standard column names (based on dataset inspection)
standard_columns = {
    'sys_updated_by', 'location', 'assignment_group', 'closed_by', 'priority',
    'caller_id', 'sys_updated_on', 'closed_at', 'assigned_to'
}

def normalize_data(example):
    """Ensures each row has the correct columns, filling missing values with 'N/A'"""
    row_data = {col: example.get(col, "N/A") for col in standard_columns}
    return row_data

# Convert to pandas to verify
#df = pd.DataFrame(normalize_data)
#print(df.head())
# Drop any extra columns that are not in the expected schema
#df = df[standard_columns]
#cleaned_dataset = Dataset.from_pandas(df)
# Apply normalization
normalized_dataset = servicenow_dataset.map(normalize_data)


# Load tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_dataset(example):
    """Formats dataset to instruction-response format"""
    instruction = example.get("question", "No question provided")
    response = example.get("answer", "No answer available")
    return {"instruction": instruction, "response": response}

# Apply formatting to both datasets
servicenow_data = normalized_dataset["train"].map(format_dataset)
#incident_data = incident_mgmt_dataset["train"].map(format_dataset)

# Merge datasets
full_dataset = servicenow_data
#+ incident_data["train"]

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",offload_folder="offload")
# Ensure that the model is first on CPU
model = model.cpu()
model = model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-mistral",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="no",
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,
    push_to_hub=False
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-mistral")
tokenizer.save_pretrained("./fine-tuned-mistral")

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-mistral")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-mistral")

# Define chatbot response function
def generate_response(user_input):
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_length=200, do_sample=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

#################
# Title of the web app
st.title("🤖 Chatbot Support")

# Load the chatbot model (cache to avoid reloading)

@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="facebook/blenderbot-400M-distill")

#chatbot = load_chatbot()

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
    response = generate_response(user_input)
    bot_response = response[0]["generated_text"]

    # Save chat in database
    DB.save_chat_to_db(user_input, bot_response)

    # Display response
    st.write("🤖 Chatbot:", bot_response)

