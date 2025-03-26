import psycopg2
from datetime import datetime

class DB():

    # Database Connection
    DB_PARAMS = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "123456",
        "host": "localhost",
        "port": "5432",
    }

    # Function to connect to PostgreSQL
    def connect_db():
        return psycopg2.connect(**DB.DB_PARAMS)

    # Function to save chat history
    def save_chat_to_db(user_msg, bot_msg):
        conn = DB.connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_message, bot_response) VALUES (%s, %s)",
            (user_msg, bot_msg),
        )
        conn.commit()
        cursor.close()
        conn.close()

    # Function to retrieve chat history
    def get_chat_history():
        conn = DB.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT user_message, bot_response, timestamp FROM chat_history ORDER BY timestamp DESC LIMIT 10")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    
    # Function to clear chat history from the database
    def clear_chat_history():
        conn = DB.connect_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history")
        conn.commit()
        cursor.close()
        conn.close()