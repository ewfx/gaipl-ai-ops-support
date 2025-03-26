import requests
import sqlite3
import smtplib
import time
import logging
from email.mime.text import MIMEText
from transformers import pipeline
import random
import psycopg2

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "facebook/bart-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

logging.basicConfig(level=logging.INFO)  # Set logging level

from psycopg2 import pool



# Mock API Calls for Prometheus, Elastic APM, and Telemetry
def get_logs():
    return [
        {"source": "prometheus", "message": "CPU usage high at 95%"},
        {"source": "elastic_apm", "message": "Database connection timeout error"},
        {"source": "telemetry", "message": "Memory leak detected in service A"}
    ]

# Hugging Face NLP model to analyze logs
nlp = pipeline("text-classification", model="facebook/bart-large-mnli")
#resolution_model = pipeline("text-generation", model="facebook/bart-large-cnn")



def analyze_log(log):
    categories = ["error", "warning", "info"]
    result = nlp(log["message"])[0]
    log["category"] = categories[random.randint(0, 2)]
    return log

# Get resolution from Hugging Face AI
def get_resolution(log):     
        API_KEY = "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"

        # API URL (replace "mistralai/Mistral-7B-Instruct-v0.1" with your model)
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

        # Headers with API key
        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Define input prompt
        payload = {
            "inputs": log['message'],
            "parameters": {"max_length": 100}
        }

        # Make API request
        response = requests.post(API_URL, json=payload, headers=headers)
        logging.info(response.json())

        return response.json()

# Incident Report Generation
def create_incident(log):
    if log["category"] == "error":
        resolution = get_resolution(log)
        return {
            "title": f"Incident: {log['message']}",
            "description": f"Source: {log['source']}\nIssue: {log['message']}\nResolution: {resolution}\nSeverity: High",
            "status": "Open"
        }
    return None

# Send Email Notification
def send_email(incident):
    sender = "your_email@example.com"
    recipient = "admin@example.com"
    msg = MIMEText(incident["description"])
    msg["Subject"] = incident["title"]
    msg["From"] = sender
    msg["To"] = recipient

    with smtplib.SMTP("smtp.example.com", 587) as server:
        server.starttls()
        server.login(sender, "your_password")
        server.sendmail(sender, recipient, msg.as_string())

# Save to Database
def save_to_db(incident):
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='admin',
        host='localhost',
        port='5432'
    )
    
    cursor = conn.cursor()
    
   # cursor.execute("INSERT INTO incidents (record_type, title, description, status) VALUES (?, ?, ?, ?)",
                   #('incident', incident["title"], incident["description"], incident["status"]))
                   
    insert_query = """
    INSERT INTO incidents (record_type, description, status)
    VALUES (%s, %s, %s)
    """
    
    # Data to insert
    data = ('value1', 'value2', 'value3')

    # Execute query
    cursor.execute(insert_query, data)
               
    conn.commit()
    conn.close()

# Automated Agent Execution
def run_agent(interval=60):
    while True:
        logs = get_logs()
        for log in logs:
            analyzed_log = analyze_log(log)
           # print(f"{analyze_loglyzed_log")
            incident = create_incident(analyzed_log)
            if incident:
                save_to_db(incident)
                send_email(incident)                
                print(f"Incident Created: {incident['title']}")
        time.sleep(interval)

# Run the agent every 60 seconds
run_agent()
