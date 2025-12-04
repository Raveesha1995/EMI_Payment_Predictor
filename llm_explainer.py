"""
LLM integration for explaining predictions and generating insights
Core feature of the EMI Payment Predictor system
"""
import os
from typing import Dict, Optional
from config import OPENAI_API_KEY, OPENAI_MODEL

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    raise ImportError("OpenAI library is required. Install it with: pip install openai")


class LLMExplainer:
    """LLM-powered explanation engine - Core feature for intelligent predictions"""
    
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        if not OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not found. LLM is a core feature of this system.")
            print("Please set OPENAI_API_KEY in .env file or environment variables.")
            print("Get your API key from: https://platform.openai.com/api-keys")
            self.client = None
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            print("LLM Explainer initialized successfully with OpenAI API")
    
    def explain_prediction(self, prediction_result: Dict, customer_history: list) -> str:
        """Generate AI-powered human-readable explanation of the prediction using LLM"""
        
        if not self.client:
            return (
                "⚠️ LLM Explanation Unavailable: OpenAI API key is required for AI-powered explanations.\n"
                "Please set OPENAI_API_KEY in your .env file.\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        prompt = f"""
You are a financial analyst. Provide a SHORT explanation (2-3 sentences max) for this EMI payment prediction:

Customer: {prediction_result['customer_id']}
Next Demand: {prediction_result.get('next_demand_date', 'N/A')}
Predicted Date: {prediction_result['predicted_payment_date']}
Avg Delay: {prediction_result['average_delay']:.1f} days
Confidence: {prediction_result['confidence_score']:.0%}

Include: (1) Why this date, (2) Risk level (low/medium/high), (3) One key recommendation.
Keep it brief and actionable.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise financial analyst. Always provide brief, actionable explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API error: {e}")
            return f"⚠️ LLM Explanation Error: {str(e)}\nPlease check your OpenAI API key and connection."
    
    def _generate_simple_explanation(self, prediction_result: Dict, customer_history: list) -> str:
        """Generate explanation without LLM"""
        explanation = f"""
Prediction Explanation for Customer {prediction_result['customer_id']}:

Based on the customer's payment history:
- Total payments recorded: {prediction_result['payment_count']}
- Average delay: {prediction_result['average_delay']:.1f} days
- Predicted next payment: {prediction_result['predicted_payment_date']}
- Confidence: {prediction_result['confidence_score']:.1%}

The prediction is based on:
1. Historical payment patterns
2. Average time between payments
3. Recent payment trends
4. Payment delay patterns

"""
        return explanation
    
    def _format_history(self, history: list) -> str:
        """Format payment history for LLM prompt"""
        formatted = []
        for payment in history[-5:]:  # Last 5 payments
            date = payment.get('payment_date', 'N/A')
            delay = payment.get('delay_days', 0)
            formatted.append(f"- Date: {date}, Delay: {delay} days")
        return "\n".join(formatted)
    
    def generate_insights(self, predictions_df) -> str:
        """Generate AI-powered business insights from batch predictions using LLM"""
        if not self.client:
            return (
                "⚠️ LLM Insights Unavailable: OpenAI API key is required for AI-powered insights.\n"
                "Please set OPENAI_API_KEY in your .env file.\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        summary = f"""
Total customers analyzed: {len(predictions_df)}
Average days until payment: {predictions_df['days_until_payment'].mean():.1f}
Customers with high delay risk: {len(predictions_df[predictions_df['average_delay'] > 7])}
"""
        
        prompt = f"""
Provide SHORT business insights (3-4 sentences max) for these EMI predictions:

{summary}
Avg confidence: {predictions_df['confidence_score'].mean():.0%}
Date range: {predictions_df['predicted_payment_date'].min()} to {predictions_df['predicted_payment_date'].max()}

Include: (1) Overall risk, (2) One key trend, (3) Top action item.
Be concise.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise financial risk analyst. Always provide brief, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating insights: {e}"

