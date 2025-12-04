"""
Flask Backend API for EMI Payment Predictor
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from predictor import EMIPaymentPredictor
from llm_explainer import LLMExplainer
from config import DATA_PATH, MODEL_PATH

# Update paths to be relative to project root
MODEL_PATH = os.path.join(parent_dir, MODEL_PATH) if not os.path.isabs(MODEL_PATH) else MODEL_PATH
DATA_PATH = os.path.join(parent_dir, DATA_PATH) if not os.path.isabs(DATA_PATH) else DATA_PATH

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize predictor
predictor = None
explainer = None

def init_predictor():
    """Initialize the predictor model"""
    global predictor, explainer
    try:
        predictor = EMIPaymentPredictor()
        # Use the fixed MODEL_PATH from this module
        predictor.load_model(MODEL_PATH)
        explainer = LLMExplainer()
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict_payment():
    """Predict payment date for a customer"""
    try:
        data = request.json
        customer_id = data.get('customer_id')
        use_llm = data.get('use_llm', False)
        
        if not customer_id:
            return jsonify({'error': 'customer_id is required'}), 400
        
        if predictor is None:
            if not init_predictor():
                return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Predict
        result = predictor.predict_next_payment_date(customer_id, DATA_PATH)
        
        # Always add LLM explanation (core feature)
        try:
            if explainer:
                explanation = explainer.explain_prediction(result, result['payment_history'])
                result['llm_explanation'] = explanation
            else:
                result['llm_explanation'] = "LLM explainer not available"
        except Exception as e:
            result['llm_explanation'] = f"LLM explanation error: {str(e)}"
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict payment dates for multiple customers"""
    try:
        data = request.json
        customer_ids = data.get('customer_ids', [])
        use_llm = data.get('use_llm', False)
        
        if not customer_ids:
            # Get all customers from data
            df = pd.read_csv(DATA_PATH)
            customer_ids = df['customer_id'].unique().tolist()
        
        if predictor is None:
            if not init_predictor():
                return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Batch predict
        results_df = predictor.predict_batch(customer_ids, DATA_PATH)
        results = results_df.to_dict('records')
        
        # Always generate LLM insights (core feature)
        insights = None
        try:
            if explainer:
                insights = explainer.generate_insights(results_df)
            else:
                insights = "LLM explainer not available"
        except Exception as e:
            insights = f"LLM insights error: {str(e)}"
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results),
            'insights': insights
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customers', methods=['GET'])
def get_customers():
    """Get list of all customers"""
    try:
        df = pd.read_csv(DATA_PATH)
        customers = df['customer_id'].unique().tolist()
        
        # Get basic stats for each customer
        customer_stats = []
        for customer_id in customers:
            customer_df = df[df['customer_id'] == customer_id]
            customer_stats.append({
                'customer_id': customer_id,
                'total_payments': len(customer_df),
                'last_payment': customer_df['payment_date'].max(),
                'first_payment': customer_df['payment_date'].min()
            })
        
        return jsonify({
            'success': True,
            'customers': customer_stats,
            'total': len(customer_stats)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customer/<customer_id>/history', methods=['GET'])
def get_customer_history(customer_id):
    """Get payment history for a specific customer"""
    try:
        df = pd.read_csv(DATA_PATH)
        customer_df = df[df['customer_id'] == customer_id]
        
        if len(customer_df) == 0:
            return jsonify({'error': 'Customer not found'}), 404
        
        history = customer_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'customer_id': customer_id,
            'history': history,
            'total_payments': len(history)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the prediction model"""
    try:
        data = request.json
        data_path = data.get('data_path', DATA_PATH)
        
        global predictor
        predictor = EMIPaymentPredictor()
        metrics = predictor.train_model(data_path)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to initialize predictor on startup
    print("Initializing EMI Payment Predictor API...")
    if os.path.exists(MODEL_PATH):
        init_predictor()
        print("Model loaded successfully!")
    else:
        print("Warning: Model not found. Train the model first or use /api/train endpoint.")
    
    print("Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

