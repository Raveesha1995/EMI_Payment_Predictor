"""
Generate predictions for all customers and save to CSV
"""
import pandas as pd
from datetime import datetime
from predictor import EMIPaymentPredictor
from config import DATA_PATH, MODEL_PATH
import os

def generate_predictions_csv():
    """Generate predictions for all customers and save to CSV"""
    
    print("=" * 60)
    print("Generating EMI Payment Predictions CSV")
    print("=" * 60)
    
    # Load data to get all customers
    print("\nLoading customer data...")
    df = pd.read_csv(DATA_PATH)
    customer_ids = df['customer_id'].unique().tolist()
    print(f"Found {len(customer_ids)} customers")
    
    # Initialize predictor
    print("\nLoading prediction model...")
    predictor = EMIPaymentPredictor()
    predictor.load_model()
    print("Model loaded successfully!")
    
    # Generate predictions for all customers
    print("\nGenerating predictions...")
    results = []
    
    for i, customer_id in enumerate(customer_ids, 1):
        try:
            result = predictor.predict_next_payment_date(customer_id, DATA_PATH)
            
            # Extract the data with proper column names
            prediction_data = {
                'Customer ID': result['customer_id'],
                'Last Demand Date': result.get('last_demand_date', '') if result.get('last_demand_date') else '',
                'Last Payment': result['last_payment_date'],
                'Next Demand Date': result.get('next_demand_date', '') if result.get('next_demand_date') else '',
                'Predicted Date': result['predicted_payment_date'],
                'Avg Delay': round(result['average_delay'], 2),
                'Confidence': f"{result['confidence_score']*100:.1f}%"
            }
            
            results.append(prediction_data)
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(customer_ids)} customers...")
                
        except Exception as e:
            print(f"  Warning: Could not predict for {customer_id}: {str(e)}")
            # Skip customers that can't be predicted
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Customer ID for better organization
    results_df = results_df.sort_values('Customer ID').reset_index(drop=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"emi_predictions_{timestamp}.csv"
    
    # Save to CSV with proper formatting
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n" + "=" * 60)
    print("Predictions Generated Successfully!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print(f"Total predictions: {len(results_df)}")
    print(f"\nColumns in output:")
    for col in results_df.columns:
        print(f"  - {col}")
    
    print(f"\nSample data (first 5 rows):")
    print(results_df.head().to_string(index=False))
    
    print(f"\n" + "=" * 60)
    print("File saved successfully!")
    print("=" * 60)
    
    return output_file, results_df

if __name__ == "__main__":
    try:
        output_file, results_df = generate_predictions_csv()
        print(f"\nYou can now open '{output_file}' in Excel to view all predictions.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
