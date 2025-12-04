"""
Main script for EMI Payment Predictor
"""
import pandas as pd
import argparse
import os
from datetime import datetime
from predictor import EMIPaymentPredictor
from llm_explainer import LLMExplainer
from config import DATA_PATH, MODEL_PATH


def train_model(data_path: str):
    """Train the prediction model"""
    print("=" * 60)
    print("EMI Payment Predictor - Model Training")
    print("=" * 60)
    
    predictor = EMIPaymentPredictor()
    metrics = predictor.train_model(data_path)
    
    print("\nTraining completed successfully!")
    return predictor


def predict_single_customer(customer_id: str, data_path: str, use_llm: bool = False):
    """Predict payment date for a single customer"""
    print("=" * 60)
    print(f"Predicting Payment Date for Customer: {customer_id}")
    print("=" * 60)
    
    predictor = EMIPaymentPredictor()
    predictor.load_model()
    
    result = predictor.predict_next_payment_date(customer_id, data_path)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Customer ID: {result['customer_id']}")
    print(f"Last Payment Date: {result['last_payment_date']}")
    print(f"Predicted Next Payment Date: {result['predicted_payment_date']}")
    print(f"Days Until Payment: {result['days_until_payment']} days")
    print(f"Average Delay: {result['average_delay']:.2f} days")
    print(f"Confidence Score: {result['confidence_score']:.1%}")
    print(f"Total Payment Records: {result['payment_count']}")
    
    if use_llm:
        print("\n" + "=" * 60)
        print("LLM EXPLANATION")
        print("=" * 60)
        explainer = LLMExplainer()
        explanation = explainer.explain_prediction(result, result['payment_history'])
        print(explanation)
    
    return result


def predict_batch(data_path: str, output_path: str = None, use_llm: bool = False):
    """Predict payment dates for all customers"""
    print("=" * 60)
    print("Batch Prediction for All Customers")
    print("=" * 60)
    
    # Load data to get all customer IDs
    df = pd.read_csv(data_path)
    customer_ids = df['customer_id'].unique().tolist()
    
    print(f"Found {len(customer_ids)} customers")
    
    predictor = EMIPaymentPredictor()
    predictor.load_model()
    
    results_df = predictor.predict_batch(customer_ids, data_path)
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total Predictions: {len(results_df)}")
    print(f"Average Days Until Payment: {results_df['days_until_payment'].mean():.1f}")
    print(f"Average Confidence: {results_df['confidence_score'].mean():.1%}")
    print(f"Earliest Predicted Date: {results_df['predicted_payment_date'].min()}")
    print(f"Latest Predicted Date: {results_df['predicted_payment_date'].max()}")
    
    if use_llm:
        print("\n" + "=" * 60)
        print("LLM INSIGHTS")
        print("=" * 60)
        explainer = LLMExplainer()
        insights = explainer.generate_insights(results_df)
        print(insights)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='EMI Payment Predictor')
    parser.add_argument('--mode', choices=['train', 'predict', 'batch'], 
                       default='predict', help='Operation mode')
    parser.add_argument('--data', type=str, default=DATA_PATH,
                       help='Path to EMI history data CSV')
    parser.add_argument('--customer-id', type=str,
                       help='Customer ID for single prediction')
    parser.add_argument('--output', type=str,
                       help='Output file path for batch predictions')
    parser.add_argument('--llm', action='store_true',
                       help='Use LLM for explanations (requires OpenAI API key)')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        print("Please ensure the data file exists or use --data to specify the path.")
        return
    
    if args.mode == 'train':
        train_model(args.data)
    
    elif args.mode == 'predict':
        if not args.customer_id:
            print("Error: --customer-id is required for single prediction")
            return
        predict_single_customer(args.customer_id, args.data, args.llm)
    
    elif args.mode == 'batch':
        predict_batch(args.data, args.output, args.llm)


if __name__ == "__main__":
    main()

