# AIChatBot
AI enabled chat bot

Prerequisite:
Install postgres database

table: table.sql

commands to execute:

pip install  -r requirements.txt

simple
python -m streamlit run chatbot_simple.py

advanced
python -m streamlit run chatbot_advanced.py


for incident api :
table: incident_table.sql
python -m uvicorn server:app --reload

API:
http://127.0.0.1:8000/records

http://127.0.0.1:8000/docs (Swagger UI)
http://127.0.0.1:8000/redoc (Redoc UI)


