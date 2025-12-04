"""
Quick start script to demonstrate the EMI Payment Predictor
"""
import os
import sys

def main():
    print("=" * 60)
    print("EMI Payment Predictor - Quick Start")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Generate sample data")
    print("2. Train the model")
    print("3. Make a sample prediction")
    print("\n" + "=" * 60)
    
    # Step 1: Generate sample data
    print("\n[Step 1/3] Generating sample data...")
    try:
        from generate_sample_data import generate_sample_data
        generate_sample_data(num_customers=50, payments_per_customer=12, 
                           output_file="data/emi_history.csv")
        print("✓ Sample data generated successfully!")
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return
    
    # Step 2: Train model
    print("\n[Step 2/3] Training model...")
    try:
        from predictor import EMIPaymentPredictor
        predictor = EMIPaymentPredictor()
        metrics = predictor.train_model("data/emi_history.csv")
        print("✓ Model trained successfully!")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return
    
    # Step 3: Make prediction
    print("\n[Step 3/3] Making sample prediction...")
    try:
        import pandas as pd
        df = pd.read_csv("data/emi_history.csv")
        sample_customer = df['customer_id'].iloc[0]
        
        result = predictor.predict_next_payment_date(sample_customer, "data/emi_history.csv")
        
        print("\n" + "=" * 60)
        print("SAMPLE PREDICTION RESULT")
        print("=" * 60)
        print(f"Customer ID: {result['customer_id']}")
        print(f"Last Payment: {result['last_payment_date']}")
        print(f"Predicted Next Payment: {result['predicted_payment_date']}")
        print(f"Days Until Payment: {result['days_until_payment']} days")
        print(f"Confidence: {result['confidence_score']:.1%}")
        print("=" * 60)
        print("\n✓ Quick start completed successfully!")
        print("\nYou can now use the main.py script for more predictions:")
        print("  python main.py --mode predict --customer-id CUST_0001")
        print("  python main.py --mode batch --data data/emi_history.csv")
        
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

