import sys
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# LangChain imports (handle version differences)
try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # Fallbacks for older LangChain versions
    try:
        from langchain.schema.output_parser import StrOutputParser
        from langchain.prompts import PromptTemplate
    except ImportError:
        raise ImportError(
            "LangChain imports failed. Install compatible versions:\n"
            "  pip install -U langchain langchain-google-genai google-generativeai\n"
        )

from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load Data
# Assuming train.csv is available locally as it contains the labels 'price_range'
try:
    df_train = pd.read_csv("./data/train.csv")
    X_train = df_train.drop("price_range", axis=1)
    y_train = df_train["price_range"]

    # Train the Price Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("ML Model trained successfully.")
except FileNotFoundError:
    print("Error: ./data/train.csv not found. Please ensure train.csv is in ./data for price prediction.")
    sys.exit(1)

# Load the provided test data
# Try common locations
test_paths = ["./data/test.csv", "./data/test.csv"]
df_test = None
for p in test_paths:
    try:
        df_test = pd.read_csv(p)
        print(f"Loaded test data from: {p}")
        break
    except FileNotFoundError:
        continue

if df_test is None:
    print("Error: test.csv not found in ./data/ or project root.")
    sys.exit(1)

# Prepare test data for prediction (drop 'id' as it's not a feature if present)
X_test = df_test.drop(columns=["id"], errors="ignore")

# 2. Predict Price Ranges for Test Data
df_test["predicted_price_range"] = rf_model.predict(X_test)


def get_recommendation(user_preferences: dict, llm_api_key: str):
    """
    Orchestrates the recommendation process.
    """

    # --- Step A: Hard Rule Filtering ---
    filtered_df = df_test.copy()

    # 1. Budget Filter (Mapping user budget input to price_range 0-3)
    # 0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost
    if "budget_range" in user_preferences:
        target_price = user_preferences["budget_range"]  # e.g., [2, 3]
        filtered_df = filtered_df[filtered_df["predicted_price_range"].isin(target_price)]

    # 2. 4G Capability (dataset has 'four_g' and 'three_g')
    if user_preferences.get("requires_4g"):
        if "four_g" not in filtered_df.columns:
            return {"error": "Dataset missing 'four_g' column for 4G filtering."}
        filtered_df = filtered_df[filtered_df["four_g"] == 1]

    # 3. Min RAM Filter
    if "min_ram" in user_preferences:
        if "ram" not in filtered_df.columns:
            return {"error": "Dataset missing 'ram' column for RAM filtering."}
        filtered_df = filtered_df[filtered_df["ram"] >= user_preferences["min_ram"]]

    # Check if we have candidates
    if filtered_df.empty:
        return {"error": "No devices found matching strict criteria."}

    # Limit to top 5 candidates to fit in LLM context window efficiently
    candidates = filtered_df.head(5).to_dict(orient="records")

    # --- Step B: LLM Hybrid Selection ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=llm_api_key)

    prompt_template = """
    You are an expert Mobile Phone Recommendation Assistant.

    User Intent: "{user_intent}"

    Here is a list of candidate devices that match the user's budget and technical constraints:
    {candidates}

    Dataset Column Guide:
    - battery_power: Energy capacity
    - clock_speed: Processor speed
    - n_cores: Number of cores
    - ram: RAM in MB
    - px_height/px_width: Resolution
    - m_dep: Mobile Depth (thickness)
    - mobile_wt: Weight

    Task:
    1. Analyze the user's intent (e.g., gaming, photography, battery life).
    2. Select the ONE best device from the candidates based on that intent.
    3. Calculate a confidence score (0.0 to 1.0).
    4. Provide reasoning linking specific specs (like RAM for gaming, Battery for travel) to the user's need.

    Output ONLY valid JSON in the following format:
    {{
      "recommendation_id": "id_of_device",
      "user_intent": "{user_intent}",
      "selected_device": "Device ID <id>",
      "confidence_score": 0.95,
      "reasoning": "Explanation here..."
    }}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["user_intent", "candidates"]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        response_str = chain.invoke({
            "user_intent": user_preferences["user_intent"],
            "candidates": json.dumps(candidates)
        })

        # Clean up json string if LLM wraps it in markdown
        response_str = response_str.strip().replace("```json", "").replace("```", "")
        return json.loads(response_str)

    except Exception as e:
        return {"error": f"LLM processing failed: {str(e)}"}


# --- Example Usage ---
if __name__ == "__main__":
    # Example User Input
    user_input = {
        "budget_range": [2, 3],       # User can afford High/Very High cost
        "requires_4g": True,
        "min_ram": 3000,              # Wants at least 3GB RAM
        "user_intent": "I play heavy games like PUBG and need a phone that won't lag. I also hate heavy phones."
    }

    # Avoid hardcoding API keys; use env vars in practice
    # e.g., export GOOGLE_API_KEY=...
    api_key = ""
    result = get_recommendation(user_input, api_key)
    print(json.dumps(result, indent=2))