"""
Data processing and feature engineering module for EMI payment prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class EMIDataProcessor:
    """Process and engineer features from EMI payment history"""
    
    def __init__(self):
        self.feature_windows = [7, 14, 30, 60, 90]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load EMI payment history data"""
        df = pd.read_csv(file_path)
        df['payment_date'] = pd.to_datetime(df['payment_date'])
        df = df.sort_values('payment_date')
        return df
    
    def calculate_payment_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate days between scheduled and actual payment dates"""
        if 'scheduled_date' in df.columns:
            # Ensure scheduled_date is datetime
            df['scheduled_date'] = pd.to_datetime(df['scheduled_date'])
            df['delay_days'] = (df['payment_date'] - df['scheduled_date']).dt.days
        else:
            # If no scheduled date, calculate days between payments
            df['delay_days'] = df['payment_date'].diff().dt.days
            df['delay_days'].fillna(0, inplace=True)
        return df
    
    def engineer_features(self, df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
        """Engineer features for a specific customer"""
        customer_df = df[df['customer_id'] == customer_id].copy()
        
        if len(customer_df) < 2:
            return None
        
        features = {}
        
        # Basic statistics
        features['total_payments'] = len(customer_df)
        features['avg_delay'] = customer_df['delay_days'].mean()
        features['std_delay'] = customer_df['delay_days'].std()
        features['max_delay'] = customer_df['delay_days'].max()
        features['min_delay'] = customer_df['delay_days'].min()
        
        # Recent payment trends
        recent_payments = customer_df.tail(3)
        features['recent_avg_delay'] = recent_payments['delay_days'].mean()
        features['recent_trend'] = recent_payments['delay_days'].diff().mean()
        
        # Payment frequency
        if len(customer_df) > 1:
            payment_intervals = customer_df['payment_date'].diff().dt.days.dropna()
            features['avg_interval'] = payment_intervals.mean()
            features['interval_std'] = payment_intervals.std()
        
        # Day of week patterns
        customer_df['day_of_week'] = customer_df['payment_date'].dt.dayofweek
        features['preferred_day'] = customer_df['day_of_week'].mode()[0] if len(customer_df['day_of_week'].mode()) > 0 else 0
        
        # Month patterns
        customer_df['month'] = customer_df['payment_date'].dt.month
        features['preferred_month'] = customer_df['month'].mode()[0] if len(customer_df['month'].mode()) > 0 else 0
        
        # Rolling statistics
        for window in self.feature_windows:
            window_df = customer_df[customer_df['payment_date'] >= 
                                   customer_df['payment_date'].max() - timedelta(days=window)]
            if len(window_df) > 0:
                features[f'delay_mean_{window}d'] = window_df['delay_days'].mean()
                features[f'payment_count_{window}d'] = len(window_df)
            else:
                features[f'delay_mean_{window}d'] = 0
                features[f'payment_count_{window}d'] = 0
        
        # Last payment information
        last_payment = customer_df.iloc[-1]
        features['last_payment_date'] = last_payment['payment_date']
        features['last_delay'] = last_payment['delay_days']
        features['days_since_last_payment'] = (datetime.now() - last_payment['payment_date']).days
        
        # Payment amount statistics (if available)
        if 'amount' in customer_df.columns:
            features['avg_amount'] = customer_df['amount'].mean()
            features['last_amount'] = last_payment['amount']
        
        return pd.DataFrame([features])
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare training data with features and targets"""
        X_list = []
        y_list = []
        
        for customer_id in df['customer_id'].unique():
            customer_df = df[df['customer_id'] == customer_id].copy()
            
            if len(customer_df) < 3:
                continue
            
            # For each payment (except the last one), use previous payments to predict next payment date
            for i in range(2, len(customer_df)):
                historical = customer_df.iloc[:i]
                target = customer_df.iloc[i]
                
                features = self.engineer_features(
                    df[df['customer_id'] == customer_id].iloc[:i], 
                    customer_id
                )
                
                if features is not None:
                    # Target: days until next payment
                    if i < len(customer_df) - 1:
                        next_payment = customer_df.iloc[i+1]
                        days_until_next = (next_payment['payment_date'] - target['payment_date']).days
                    else:
                        days_until_next = (target['payment_date'] - historical.iloc[-1]['payment_date']).days
                    
                    X_list.append(features)
                    y_list.append(days_until_next)
        
        if len(X_list) == 0:
            return None, None
        
        X = pd.concat(X_list, ignore_index=True)
        y = np.array(y_list)
        
        # Drop non-numeric columns for model training
        X_numeric = X.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.drop(columns=['last_payment_date'], errors='ignore')
        
        return X_numeric, y

