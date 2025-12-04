"""
Main prediction module using ML models and LLM for EMI payment date prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from data_processor import EMIDataProcessor
from config import MODEL_PATH, MIN_HISTORY_RECORDS


class EMIPaymentPredictor:
    """Predict next EMI payment date based on historical data"""
    
    def __init__(self):
        self.processor = EMIDataProcessor()
        self.model = None
        self.feature_names = None
        
    def train_model(self, data_path: str) -> Dict:
        """Train the prediction model"""
        print("Loading and processing data...")
        df = self.processor.load_data(data_path)
        df = self.processor.calculate_payment_delays(df)
        
        print("Engineering features...")
        X, y = self.processor.prepare_training_data(df)
        
        if X is None or len(X) == 0:
            raise ValueError("Insufficient data for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        print("Training model...")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.feature_names = X.columns.tolist()
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        print(f"\nModel Performance:")
        print(f"Train MAE: {train_mae:.2f} days")
        print(f"Test MAE: {test_rmse:.2f} days")
        print(f"Train RMSE: {train_rmse:.2f} days")
        print(f"Test RMSE: {test_rmse:.2f} days")
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, MODEL_PATH)
        
        print(f"\nModel saved to {MODEL_PATH}")
        
        return metrics
    
    def load_model(self, model_path: Optional[str] = None):
        """Load trained model"""
        # Use provided path or default from config
        path_to_use = model_path or MODEL_PATH
        
        # If relative path, try to resolve from current working directory or script location
        if not os.path.isabs(path_to_use):
            # Try current directory first
            if os.path.exists(path_to_use):
                path_to_use = os.path.abspath(path_to_use)
            else:
                # Try from project root (parent of current file)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = script_dir
                potential_path = os.path.join(project_root, path_to_use)
                if os.path.exists(potential_path):
                    path_to_use = potential_path
                else:
                    # Try from parent directory (in case running from backend/)
                    parent_path = os.path.join(os.path.dirname(script_dir), path_to_use)
                    if os.path.exists(parent_path):
                        path_to_use = parent_path
        
        if not os.path.exists(path_to_use):
            raise FileNotFoundError(f"Model not found at {path_to_use}. Please train the model first.")
        
        model_data = joblib.load(path_to_use)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        print("Model loaded successfully")
    
    def predict_next_payment_date(self, customer_id: str, data_path: str) -> Dict:
        """Predict next payment date for a customer"""
        if self.model is None:
            self.load_model()
        
        # Load customer data
        df = self.processor.load_data(data_path)
        df = self.processor.calculate_payment_delays(df)
        
        customer_df = df[df['customer_id'] == customer_id]
        
        if len(customer_df) < MIN_HISTORY_RECORDS:
            raise ValueError(f"Customer needs at least {MIN_HISTORY_RECORDS} payment records")
        
        # Engineer features
        features = self.processor.engineer_features(df, customer_id)
        
        if features is None:
            raise ValueError("Could not engineer features for customer")
        
        # Prepare features for prediction
        X = features.select_dtypes(include=[np.number]).copy()
        X = X.drop(columns=['last_payment_date'], errors='ignore')
        
        # Ensure feature order matches training
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        # Get customer payment history for context
        payment_history = customer_df[['payment_date', 'delay_days']].to_dict('records')
        
        # Calculate average delay
        average_delay = customer_df['delay_days'].mean()
        
        # Get last demand date (last scheduled_date) and calculate next demand date
        last_demand_date = None
        last_demand_date_str = None
        next_demand_date = None
        next_demand_date_str = None
        
        if 'scheduled_date' in customer_df.columns and len(customer_df) > 0:
            last_demand_date = customer_df['scheduled_date'].iloc[-1]
            if pd.notna(last_demand_date):
                if not isinstance(last_demand_date, pd.Timestamp):
                    last_demand_date = pd.to_datetime(last_demand_date)
                last_demand_date_str = last_demand_date.strftime('%Y-%m-%d')
                
                # Calculate next demand date: Same day of month, next month
                # EMI dates are fixed (e.g., 5th, 10th, 15th of each month)
                # Use the last scheduled date's day of month for the next month
                day_of_month = last_demand_date.day
                
                # Calculate next month
                if last_demand_date.month == 12:
                    next_year = last_demand_date.year + 1
                    next_month = 1
                else:
                    next_year = last_demand_date.year
                    next_month = last_demand_date.month + 1
                
                # Handle cases where day doesn't exist in next month (e.g., Jan 31 -> Feb 28/29)
                try:
                    next_demand_date = pd.Timestamp(year=next_year, month=next_month, day=day_of_month)
                except ValueError:
                    # If day doesn't exist (e.g., Feb 31), use last day of month
                    from calendar import monthrange
                    last_day = monthrange(next_year, next_month)[1]
                    next_demand_date = pd.Timestamp(year=next_year, month=next_month, day=min(day_of_month, last_day))
                
                next_demand_date_str = next_demand_date.strftime('%Y-%m-%d')
        
        # Calculate predicted date: Next Demand Date + Average Delay
        # This makes business sense - if customer pays 4 days late on average,
        # they'll pay 4 days after the demand date
        if next_demand_date is not None:
            # Round average delay to nearest integer for date calculation
            delay_days = int(round(average_delay))
            predicted_date = next_demand_date + timedelta(days=delay_days)
        else:
            # Fallback: Use ML model prediction if no demand date available
            days_until_payment = self.model.predict(X)[0]
            days_until_payment = max(0, int(round(days_until_payment)))
            last_payment_date = features['last_payment_date'].iloc[0]
            if isinstance(last_payment_date, pd.Timestamp):
                predicted_date = last_payment_date + timedelta(days=int(days_until_payment))
            else:
                predicted_date = pd.to_datetime(last_payment_date) + timedelta(days=int(days_until_payment))
        
        # Calculate days until payment for display
        last_payment_date = features['last_payment_date'].iloc[0]
        if isinstance(last_payment_date, pd.Timestamp):
            days_until_payment = (predicted_date - last_payment_date).days
        else:
            days_until_payment = (predicted_date - pd.to_datetime(last_payment_date)).days
        
        result = {
            'customer_id': customer_id,
            'predicted_payment_date': predicted_date.strftime('%Y-%m-%d'),
            'days_until_payment': days_until_payment,
            'last_payment_date': last_payment_date.strftime('%Y-%m-%d') if hasattr(last_payment_date, 'strftime') else str(last_payment_date),
            'last_demand_date': last_demand_date_str,
            'next_demand_date': next_demand_date_str,
            'confidence_score': self._calculate_confidence(features),
            'payment_history': payment_history[-5:],  # Last 5 payments
            'average_delay': average_delay,
            'payment_count': len(customer_df)
        }
        
        return result
    
    def _calculate_confidence(self, features: pd.DataFrame) -> float:
        """Calculate confidence score based on data quality"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence with more data
        if features['total_payments'].iloc[0] > 10:
            confidence += 0.1
        elif features['total_payments'].iloc[0] > 5:
            confidence += 0.05
        
        # Decrease confidence with high variance
        if features['std_delay'].iloc[0] > 10:
            confidence -= 0.1
        elif features['std_delay'].iloc[0] < 3:
            confidence += 0.1
        
        return min(0.95, max(0.5, confidence))
    
    def predict_batch(self, customer_ids: list, data_path: str) -> pd.DataFrame:
        """Predict for multiple customers"""
        results = []
        
        for customer_id in customer_ids:
            try:
                result = self.predict_next_payment_date(customer_id, data_path)
                results.append(result)
            except Exception as e:
                print(f"Error predicting for customer {customer_id}: {str(e)}")
                continue
        
        return pd.DataFrame(results)

