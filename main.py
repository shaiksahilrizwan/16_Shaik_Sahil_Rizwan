from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from model import get_device_recommendation

# Create the App
app = FastAPI(title="Device Recommendation API")

# Define Input Schema using Pydantic
class UserInput(BaseModel):
    budget_range: List[int] = Field(..., description="List of price ranges e.g. [0, 1, 2, 3]")
    requires_4g: bool = Field(False, description="Whether 4G is required")
    min_ram: int = Field(0, description="Minimum RAM in MB")
    user_intent: str = Field(..., description="Text description of user needs")

    class Config:
        json_schema_extra = {
            "example": {
                "budget_range": [2, 3],
                "requires_4g": True,
                "min_ram": 3000,
                "user_intent": "I need a gaming phone that is lightweight."
            }
        }

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/recommend")
def recommend_device(input_data: UserInput):
    """
    Receives user constraints and returns a device recommendation.
    """
    # Convert Pydantic model to Python dict
    input_dict = input_data.dict()
    
    # Call the function from model.py
    result = get_device_recommendation(input_dict)
    
    # Check for errors returned by the model
    if "error" in result:
        # If it's a 'No devices found' error, strictly speaking it's a 404 or 400, 
        # but 200 with error info is often easier for frontends.
        # Here we raise HTTP exception for clarity.
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

# To run locally: uvicorn main:app --reload