"""
Generate sample EMI payment history data for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(num_customers: int = 50, payments_per_customer: int = 12, 
                        output_file: str = "data/emi_history.csv"):
    """Generate sample EMI payment data"""
    
    print(f"Generating sample data for {num_customers} customers...")
    
    data = []
    
    for customer_id in range(1, num_customers + 1):
        # Base payment date (start from 6 months ago)
        base_date = datetime.now() - timedelta(days=180)
        
        # Customer-specific payment behavior
        base_interval = random.choice([28, 30, 31, 32])  # Monthly payment cycle
        delay_tendency = random.choice(['on_time', 'early', 'late'])
        
        if delay_tendency == 'on_time':
            delay_range = (-2, 2)
        elif delay_tendency == 'early':
            delay_range = (-5, 0)
        else:  # late
            delay_range = (0, 10)
        
        current_date = base_date
        
        for payment_num in range(payments_per_customer):
            # Add some randomness to payment interval
            interval = base_interval + random.randint(-3, 3)
            current_date = current_date + timedelta(days=interval)
            
            # Calculate delay
            delay = random.randint(*delay_range)
            actual_payment_date = current_date + timedelta(days=delay)
            
            # Skip if payment date is in the future
            if actual_payment_date > datetime.now():
                break
            
            # Generate payment amount (EMI amount)
            amount = random.uniform(5000, 50000)
            
            data.append({
                'customer_id': f'CUST_{customer_id:04d}',
                'payment_date': actual_payment_date.strftime('%Y-%m-%d'),
                'scheduled_date': current_date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'payment_status': 'completed'
            })
    
    df = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Sample data generated: {len(df)} payment records")
    print(f"Saved to: {output_file}")
    print(f"\nData preview:")
    print(df.head(10))
    print(f"\nData statistics:")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Date range: {df['payment_date'].min()} to {df['payment_date'].max()}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample EMI payment data')
    parser.add_argument('--customers', type=int, default=50,
                       help='Number of customers')
    parser.add_argument('--payments', type=int, default=12,
                       help='Payments per customer')
    parser.add_argument('--output', type=str, default='data/emi_history.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    generate_sample_data(args.customers, args.payments, args.output)

