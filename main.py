from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
import os

# --- FIX: Import from the file inside the folder ---
# Syntax is: from folder_name.file_name import function_name
from model.model import get_device_recommendation

# Create the App
app = FastAPI(title="Device Recommendation API")

# --- Input Schema ---
class UserInput(BaseModel):
    budget_range: List[int] = Field(..., description="List of price ranges e.g. [0, 1, 2, 3]")
    requires_4g: bool = Field(False, description="Whether 4G is required")
    min_ram: int = Field(0, description="Minimum RAM in MB")
    user_intent: str = Field(..., description="Text description of user needs")

# --- API Routes ---

@app.post("/api/recommend")
async def recommend_device(input_data: UserInput):
    """
    API Endpoint: Receives user constraints -> Returns JSON Recommendation
    """
    # Convert Pydantic model to Python dict
    input_dict = input_data.dict()
    
    # Call your ML/LLM function
    result = get_device_recommendation(input_dict)
    
    # Error Handling
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

# --- Frontend Serving ---
# This must be placed AFTER the API routes so it doesn't overwrite them.
# 'html=True' means it will serve index.html automatically at the root URL (/)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Run with: uvicorn main:app --reload