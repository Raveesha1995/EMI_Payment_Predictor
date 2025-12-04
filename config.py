"""
Configuration file for EMI Payment Predictor
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Configuration (Required - LLM is a core feature)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")  # Default to GPT-4 Turbo

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found. LLM features require an API key.")
    print("Set it in .env file or environment variables.")
    print("Get your API key from: https://platform.openai.com/api-keys")
else:
    print(f"OpenAI API key loaded. Using model: {OPENAI_MODEL}")

# Model Configuration
MODEL_PATH = "models/emi_predictor_model.pkl"
DATA_PATH = "data/emi_history.csv"

# Prediction Settings
MIN_HISTORY_RECORDS = 3  # Minimum records needed for prediction
DEFAULT_PREDICTION_DAYS = 30  # Default prediction window

# Feature Engineering Settings
FEATURE_WINDOWS = [7, 14, 30, 60, 90]  # Days for rolling features

