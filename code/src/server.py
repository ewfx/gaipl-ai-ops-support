from fastapi import FastAPI
import asyncpg
import os
from fastapi.middleware.cors import CORSMiddleware
from database_conn import DB

# FastAPI app instance
app = FastAPI()

# Allow CORS for React frontend (Adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to allow specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL Database Config
DB_CONFIG = {
    "user": "postgres",
    "password": "123456",
    "database": "postgres",
    "host": "localhost",  # Change if running on a remote server
    "port": 5432  # Default PostgreSQL port
}

# Function to fetch service_now_records from PostgreSQL
async def fetch_records():
    conn = await asyncpg.connect(**DB_CONFIG)
    records = await conn.fetch("SELECT * FROM incidents")
    await conn.close()
    return [dict(record) for record in records]  # Convert records to list of dictionaries

# API Route to fetch service_now_records
@app.get("/records")
async def get_records():
    records = await fetch_records()
    return {"data": records}

# Run the API server using: uvicorn server:app --reload
