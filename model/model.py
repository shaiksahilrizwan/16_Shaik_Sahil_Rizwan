import sys
import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Imports & Dependency Checks 
try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.schema.output_parser import StrOutputParser
        from langchain.prompts import PromptTemplate
    except ImportError:
        raise ImportError(
            "LangChain imports failed. Install compatible versions:\n"
            "pip install -U langchain langchain-google-genai google-generativeai"
        )

from langchain_google_genai import ChatGoogleGenerativeAI

# Global Cache (To avoid re-training on every call) ---
_MODEL_CACHE = {
    "rf_model": None,
    "processed_test_data": None
}

def get_device_recommendation(user_input: dict) -> dict:
    """
    Single function that handles ML training (cached), data loading, 
    filtering, and LLM reasoning.
    
    Args:
        user_input (dict): Dictionary containing user constraints.
                           e.g., {"budget_range": [2,3], "user_intent": "..."}
                           
    Returns:
        dict: The final recommendation in JSON format.
    """
    
    # --- CONFIGURATION (Pre-defined paths and keys) ---
    TRAIN_PATH = "model/data/train.csv"
    TEST_PATH = "model/data/test.csv"
    
    # use the env instead of this approch
    API_KEY = ""  
    
    # run once
    global _MODEL_CACHE
    
    if _MODEL_CACHE["rf_model"] is None:
        print("Initializing System: Loading Data and Training Model...")
        try:
             # Load Training Data
            if not os.path.exists(TRAIN_PATH):
                return {"error": f"Train file not found at {TRAIN_PATH}"}
            
            df_train = pd.read_csv(TRAIN_PATH)
            X_train = df_train.drop("price_range", axis=1)
            y_train = df_train["price_range"]

            # Train Model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            _MODEL_CACHE["rf_model"] = rf_model
            print("ML Model trained successfully.")

            # Load and Pre-process Test Data
            if not os.path.exists(TEST_PATH):
                return {"error": f"Test file not found at {TEST_PATH}"}
            
            df_test = pd.read_csv(TEST_PATH)
            
            # Handle ID column if present
            X_test = df_test.drop(columns=["id"], errors="ignore")
            
            # Predict Price Ranges immediately
            df_test["predicted_price_range"] = rf_model.predict(X_test)
            _MODEL_CACHE["processed_test_data"] = df_test
            
        except Exception as e:
            return {"error": f"System Initialization Failed: {str(e)}"}

    # Filter Logic
    df_processed = _MODEL_CACHE["processed_test_data"]
    filtered_df = df_processed.copy()

    # Budget Filter
    if "budget_range" in user_input:
        filtered_df = filtered_df[filtered_df["predicted_price_range"].isin(user_input["budget_range"])]

    # 4G Filter
    if user_input.get("requires_4g"):
        if "four_g" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["four_g"] == 1]

    # Min RAM Filter
    if "min_ram" in user_input:
        if "ram" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["ram"] >= user_input["min_ram"]]

    # Check results
    if filtered_df.empty:
        return {"error": "No devices found matching strict criteria."}

    # Prepare candidates (Top 5 to save context window)
    candidates = filtered_df.head(5).to_dict(orient="records")

    #LLM Reasoning
    if not API_KEY or API_KEY == "YOUR_GOOGLE_GEMINI_API_KEY":
        return {"error": "Invalid API Key. Please update the API_KEY variable in the function."}

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

        prompt_template = """
        You are an expert Mobile Phone Recommendation Assistant.

        User Intent: "{user_intent}"

        Here is a list of candidate devices that match the user's budget and technical constraints:
        {candidates}

        Dataset Column Guide:
        - battery_power: Energy capacity (mAh)
        - clock_speed: Processor speed (GHz)
        - ram: RAM in MB
        - px_height/px_width: Resolution
        - mobile_wt: Weight (g)

        Task:
        1. Analyze the user's intent.
        2. Select the ONE best device from the candidates.
        3. Provide reasoning based on technical specs.

        Output ONLY valid JSON in this format:
        {{
          "recommendation_id": "rec_<device_id>",
          "user_intent": "{user_intent}",
          "selected_device": {{ "id": <id>, "name": "Device <id>" }},
          "confidence_score": 0.95,
          "reasoning": "Explanation..."
        }}
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_intent", "candidates"]
        )

        chain = prompt | llm | StrOutputParser()

        response_str = chain.invoke({
            "user_intent": user_input["user_intent"],
            "candidates": json.dumps(candidates)
        })

        # Cleaning code blocks
        cleaned_response = response_str.strip().replace("```json", "").replace("```", "")
        
        return json.loads(cleaned_response)

    except Exception as e:
        return {"error": f"LLM Processing Failed: {str(e)}"}