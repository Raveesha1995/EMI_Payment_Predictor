"""
Script to run the backend server
"""
from waitress import serve
from app import app
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from predictor import EMIPaymentPredictor
from config import MODEL_PATH

# Fix MODEL_PATH to be relative to project root
MODEL_PATH_FIXED = os.path.join(parent_dir, MODEL_PATH) if not os.path.isabs(MODEL_PATH) else MODEL_PATH

def main():
    print("=" * 60)
    print("EMI Payment Predictor - Backend Server")
    print("=" * 60)
    
    # Try to load model
    if os.path.exists(MODEL_PATH_FIXED):
        try:
            predictor = EMIPaymentPredictor()
            predictor.load_model(MODEL_PATH_FIXED)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"⚠ Warning: Could not load model: {e}")
            print("  You can train the model using the /api/train endpoint")
    else:
        print(f"⚠ Warning: Model not found at {MODEL_PATH_FIXED}")
        print("  Train the model first using: python main.py --mode train")
    
    print("\nStarting server on http://localhost:5000")
    print("API endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/predict - Predict single customer")
    print("  POST /api/predict/batch - Batch predictions")
    print("  GET  /api/customers - List all customers")
    print("  GET  /api/customer/<id>/history - Customer history")
    print("  POST /api/train - Train model")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run with waitress for production
    serve(app, host='0.0.0.0', port=5000, threads=4)

if __name__ == '__main__':
    main()

