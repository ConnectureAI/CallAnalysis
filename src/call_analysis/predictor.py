"""
Predictive analytics module for the call analysis system.

This module provides machine learning-based predictions for call volume,
sentiment trends, and operational planning to help optimize staffing and resources.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from .config import get_settings
from .models import CallInsight

logger = logging.getLogger(__name__)


class PredictiveAnalytics:
    """
    Machine learning-based predictive analytics for call center operations.
    
    This class builds and maintains prediction models for:
    - Call volume forecasting
    - Sentiment trend analysis  
    - Revenue opportunity prediction
    - Staffing optimization recommendations
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the predictive analytics system.
        
        Args:
            models_dir: Directory to store trained models. If None, uses config.
        """
        settings = get_settings()
        self.models_dir = models_dir or settings.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Training data
        self.historical_insights: List[CallInsight] = []
        self.is_trained = False
        
        # Model configuration
        self.min_training_samples = 50
        self.feature_windows = [7, 14, 30]  # Rolling window sizes
        self.prediction_horizon = 7  # Days to predict ahead
        
        # Load existing models if available
        self._load_models()
    
    def add_insights(self, insights: List[CallInsight]) -> None:
        """
        Add new insights to the historical dataset.
        
        Args:
            insights: List of CallInsight objects to add
        """
        self.historical_insights.extend(insights)
        logger.info(f"Added {len(insights)} insights. Total: {len(self.historical_insights)}")
        
        # Retrain models if we have enough data
        if len(self.historical_insights) >= self.min_training_samples:
            logger.info("Sufficient data available. Triggering model retraining.")
            self.train_models()
    
    def train_models(self) -> Dict[str, Dict[str, float]]:
        """
        Train prediction models on historical data.
        
        Returns:
            Dictionary containing training metrics for each model
        """
        if len(self.historical_insights) < self.min_training_samples:
            logger.warning(f"Insufficient data for training. Need {self.min_training_samples}, have {len(self.historical_insights)}")
            return {}
        
        logger.info("Starting model training...")
        
        # Convert insights to DataFrame
        df = self._prepare_training_data()
        
        if len(df) < 10:
            logger.warning("Insufficient processed data for training")
            return {}
        
        # Train individual models
        training_results = {}
        
        try:
            # Call volume prediction model
            volume_metrics = self._train_call_volume_model(df)
            training_results["call_volume"] = volume_metrics
            
            # Sentiment prediction model
            sentiment_metrics = self._train_sentiment_model(df)
            training_results["sentiment"] = sentiment_metrics
            
            # Revenue opportunity model
            revenue_metrics = self._train_revenue_model(df)
            training_results["revenue"] = revenue_metrics
            
            # Anomaly detection model
            anomaly_metrics = self._train_anomaly_model(df)
            training_results["anomaly"] = anomaly_metrics
            
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
        
        return training_results
    
    def predict_next_period(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate predictions for the next specified period.
        
        Args:
            days: Number of days to predict ahead
            
        Returns:
            Dictionary containing predictions and recommendations
        """
        if not self.is_trained:
            return {
                "error": "Models not trained yet. Need more historical data.",
                "required_samples": self.min_training_samples,
                "current_samples": len(self.historical_insights)
            }
        
        logger.info(f"Generating predictions for next {days} days")
        
        predictions = {
            "daily_forecasts": [],
            "weekly_totals": {},
            "recommendations": [],
            "confidence_scores": {},
            "model_performance": self._get_model_performance(),
        }
        
        # Generate daily predictions
        for i in range(days):
            future_date = datetime.now().date() + timedelta(days=i+1)
            daily_prediction = self._predict_single_day(future_date)
            predictions["daily_forecasts"].append(daily_prediction)
        
        # Calculate aggregated metrics
        predictions["weekly_totals"] = self._calculate_weekly_totals(predictions["daily_forecasts"])
        
        # Generate actionable recommendations
        predictions["recommendations"] = self._generate_recommendations(predictions)
        
        # Add confidence scores
        predictions["confidence_scores"] = self._calculate_confidence_scores(predictions)
        
        return predictions
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """
        Convert insights to structured DataFrame for model training.
        
        Returns:
            Prepared DataFrame with features and targets
        """
        # Convert insights to basic DataFrame
        data = []
        for insight in self.historical_insights:
            data.append({
                'timestamp': insight.timestamp,
                'sentiment_score': insight.sentiment_score,
                'urgency_level': insight.urgency_level,
                'revenue_opportunity': insight.revenue_opportunity,
                'confidence_score': insight.confidence_score,
                'primary_intent': insight.primary_intent,
                'resolution_status': insight.resolution_status,
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Feature engineering
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        
        # Create daily aggregations
        daily_df = df.groupby(df['timestamp'].dt.date).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'urgency_level': 'mean',
            'revenue_opportunity': 'sum',
            'confidence_score': 'mean',
            'hour': 'first',
            'day_of_week': 'first', 
            'month': 'first',
            'is_weekend': 'first',
            'is_business_hours': 'first',
        }).reset_index()
        
        # Flatten column names
        daily_df.columns = ['date'] + [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                      for col in daily_df.columns[1:]]
        
        # Rename count column to call_volume
        daily_df = daily_df.rename(columns={'sentiment_score_count': 'call_volume'})
        
        # Add rolling features for trend analysis
        for window in self.feature_windows:
            if len(daily_df) > window:
                daily_df[f'call_volume_avg_{window}d'] = daily_df['call_volume'].rolling(window).mean()
                daily_df[f'sentiment_avg_{window}d'] = daily_df['sentiment_score_mean'].rolling(window).mean()
                daily_df[f'revenue_avg_{window}d'] = daily_df['revenue_opportunity_sum'].rolling(window).mean()
        
        # Drop rows with NaN values from rolling calculations
        daily_df = daily_df.dropna()
        
        logger.info(f"Prepared training data: {len(daily_df)} daily samples")
        return daily_df
    
    def _train_call_volume_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train call volume prediction model."""
        logger.info("Training call volume prediction model")
        
        # Select features for call volume prediction
        feature_cols = [col for col in df.columns if col.endswith(('_avg_7d', '_avg_14d', '_avg_30d'))]
        feature_cols.extend(['hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours'])
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols:
            logger.warning("No suitable features found for call volume model")
            return {}
        
        X = df[feature_cols].fillna(0)
        y = df['call_volume']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and scaler
        self.models['call_volume'] = model
        self.scalers['call_volume'] = scaler
        self.model_metadata['call_volume'] = {
            'features': feature_cols,
            'mae': mae,
            'r2': r2,
            'training_samples': len(X_train),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Call volume model trained. MAE: {mae:.2f}, R²: {r2:.3f}")
        return {'mae': mae, 'r2': r2}
    
    def _train_sentiment_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train sentiment prediction model."""
        logger.info("Training sentiment prediction model")
        
        feature_cols = [col for col in df.columns if col.endswith(('_avg_7d', '_avg_14d', '_avg_30d'))]
        feature_cols.extend(['hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours'])
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols or 'sentiment_score_mean' not in df.columns:
            logger.warning("No suitable features found for sentiment model")
            return {}
        
        X = df[feature_cols].fillna(0)
        y = df['sentiment_score_mean']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store
        self.models['sentiment'] = model
        self.scalers['sentiment'] = scaler
        self.model_metadata['sentiment'] = {
            'features': feature_cols,
            'mae': mae,
            'r2': r2,
            'training_samples': len(X_train),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Sentiment model trained. MAE: {mae:.3f}, R²: {r2:.3f}")
        return {'mae': mae, 'r2': r2}
    
    def _train_revenue_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train revenue opportunity prediction model."""
        logger.info("Training revenue prediction model")
        
        feature_cols = [col for col in df.columns if col.endswith(('_avg_7d', '_avg_14d', '_avg_30d'))]
        feature_cols.extend(['hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours'])
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols or 'revenue_opportunity_sum' not in df.columns:
            logger.warning("No suitable features found for revenue model")
            return {}
        
        X = df[feature_cols].fillna(0)
        y = df['revenue_opportunity_sum']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store
        self.models['revenue'] = model
        self.scalers['revenue'] = scaler
        self.model_metadata['revenue'] = {
            'features': feature_cols,
            'mae': mae,
            'r2': r2,
            'training_samples': len(X_train),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Revenue model trained. MAE: ${mae:.0f}, R²: {r2:.3f}")
        return {'mae': mae, 'r2': r2}
    
    def _train_anomaly_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train anomaly detection model."""
        logger.info("Training anomaly detection model")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['date']]
        
        if len(numeric_cols) < 3:
            logger.warning("Insufficient numeric features for anomaly detection")
            return {}
        
        X = df[numeric_cols].fillna(0)
        
        # Train isolation forest
        model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)
        
        # Calculate performance metrics
        anomaly_scores = model.decision_function(X)
        outliers = model.predict(X)
        outlier_ratio = np.sum(outliers == -1) / len(outliers)
        
        # Store
        self.models['anomaly'] = model
        self.model_metadata['anomaly'] = {
            'features': numeric_cols,
            'outlier_ratio': outlier_ratio,
            'training_samples': len(X),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Anomaly model trained. Outlier ratio: {outlier_ratio:.3f}")
        return {'outlier_ratio': outlier_ratio}
    
    def _predict_single_day(self, date) -> Dict[str, Any]:
        """Generate prediction for a single day."""
        # Create feature vector for the date
        dt = datetime.combine(date, datetime.min.time())
        base_features = {
            'hour': 12,  # Noon as default
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_business_hours': 1,
        }
        
        # Add rolling average features (use recent historical values)
        if self.historical_insights:
            recent_insights = self.historical_insights[-30:]  # Last 30 insights
            if recent_insights:
                avg_sentiment = np.mean([i.sentiment_score for i in recent_insights])
                avg_revenue = np.mean([i.revenue_opportunity for i in recent_insights])
                
                for window in self.feature_windows:
                    base_features[f'sentiment_avg_{window}d'] = avg_sentiment
                    base_features[f'revenue_avg_{window}d'] = avg_revenue
                    base_features[f'call_volume_avg_{window}d'] = len(recent_insights) / 7  # Rough estimate
        
        prediction = {'date': date.isoformat()}
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            if model_name == 'anomaly':
                continue  # Skip anomaly model for daily predictions
                
            try:
                metadata = self.model_metadata[model_name]
                features = metadata['features']
                
                # Create feature vector
                feature_vector = np.array([[base_features.get(f, 0) for f in features]])
                
                # Scale and predict
                if model_name in self.scalers:
                    feature_vector = self.scalers[model_name].transform(feature_vector)
                
                pred_value = model.predict(feature_vector)[0]
                
                # Post-process predictions
                if model_name == 'call_volume':
                    pred_value = max(0, int(round(pred_value)))
                elif model_name == 'sentiment':
                    pred_value = np.clip(pred_value, -1, 1)
                elif model_name == 'revenue':
                    pred_value = max(0, pred_value)
                
                prediction[model_name] = pred_value
                
            except Exception as e:
                logger.error(f"Error predicting {model_name} for {date}: {e}")
                prediction[model_name] = 0
        
        return prediction
    
    def _calculate_weekly_totals(self, daily_forecasts: List[Dict]) -> Dict[str, Any]:
        """Calculate weekly aggregate metrics from daily forecasts."""
        if not daily_forecasts:
            return {}
        
        totals = {
            'total_calls': sum(day.get('call_volume', 0) for day in daily_forecasts),
            'avg_sentiment': np.mean([day.get('sentiment', 0) for day in daily_forecasts]),
            'total_revenue_opportunity': sum(day.get('revenue', 0) for day in daily_forecasts),
            'prediction_period': f"{daily_forecasts[0]['date']} to {daily_forecasts[-1]['date']}"
        }
        
        return totals
    
    def _generate_recommendations(self, predictions: Dict) -> List[str]:
        """Generate actionable business recommendations from predictions."""
        recommendations = []
        
        daily_forecasts = predictions.get('daily_forecasts', [])
        weekly_totals = predictions.get('weekly_totals', {})
        
        if not daily_forecasts:
            return recommendations
        
        # Call volume recommendations
        weekly_calls = weekly_totals.get('total_calls', 0)
        if weekly_calls > 0:
            daily_avg = weekly_calls / len(daily_forecasts)
            
            # High volume days
            high_volume_days = [
                day for day in daily_forecasts 
                if day.get('call_volume', 0) > daily_avg * 1.3
            ]
            
            if high_volume_days:
                peak_day = max(high_volume_days, key=lambda x: x.get('call_volume', 0))
                recommendations.append(
                    f"📈 HIGH VOLUME ALERT: Expected {peak_day['call_volume']} calls on {peak_day['date']}. "
                    f"Schedule additional reception staff (+{((peak_day['call_volume'] / daily_avg - 1) * 100):.0f}% above average)."
                )
            
            # Low volume opportunities
            low_volume_days = [
                day for day in daily_forecasts 
                if day.get('call_volume', 0) < daily_avg * 0.7
            ]
            
            if low_volume_days:
                recommendations.append(
                    f"📊 MARKETING OPPORTUNITY: {len(low_volume_days)} days with lower call volume predicted. "
                    f"Consider promotional campaigns or outreach activities."
                )
        
        # Sentiment recommendations
        avg_sentiment = weekly_totals.get('avg_sentiment', 0)
        if avg_sentiment < -0.1:
            recommendations.append(
                f"😕 QUALITY CONCERN: Below-average customer satisfaction predicted (score: {avg_sentiment:.2f}). "
                f"Review staff training and customer service procedures."
            )
        elif avg_sentiment > 0.5:
            recommendations.append(
                f"😊 EXCELLENT OUTLOOK: High customer satisfaction predicted (score: {avg_sentiment:.2f}). "
                f"Great opportunity to gather testimonials and referrals."
            )
        
        # Revenue recommendations
        total_revenue = weekly_totals.get('total_revenue_opportunity', 0)
        if total_revenue > 10000:
            recommendations.append(
                f"💰 REVENUE OPPORTUNITY: ${total_revenue:,.0f} in potential revenue identified. "
                f"Focus on conversion optimization and follow-up procedures."
            )
        
        return recommendations
    
    def _calculate_confidence_scores(self, predictions: Dict) -> Dict[str, float]:
        """Calculate confidence scores for predictions."""
        confidence_scores = {}
        
        # Base confidence on model performance and data quality
        for model_name, metadata in self.model_metadata.items():
            if model_name == 'anomaly':
                continue
                
            # Calculate confidence based on R² score and sample size  
            r2 = metadata.get('r2', 0)
            sample_size = metadata.get('training_samples', 0)
            
            # Confidence increases with R² and sample size
            size_factor = min(sample_size / 1000, 1.0)  # Max out at 1000 samples
            confidence = (r2 + size_factor) / 2
            confidence = max(0.1, min(0.9, confidence))  # Bound between 0.1 and 0.9
            
            confidence_scores[model_name] = confidence
        
        return confidence_scores
    
    def _get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all trained models."""
        return {
            model_name: {
                'mae': metadata.get('mae'),
                'r2': metadata.get('r2'),
                'training_samples': metadata.get('training_samples'),
                'trained_at': metadata.get('trained_at'),
                'features_count': len(metadata.get('features', []))
            }
            for model_name, metadata in self.model_metadata.items()
        }
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            # Save models
            for model_name, model in self.models.items():
                model_path = self.models_dir / f"{model_name}_model.joblib"
                joblib.dump(model, model_path)
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = self.models_dir / f"{scaler_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata_path = self.models_dir / "model_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.model_metadata, f)
            
            logger.info(f"Models saved to {self.models_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        try:
            metadata_path = self.models_dir / "model_metadata.pkl"
            if not metadata_path.exists():
                logger.info("No existing models found")
                return
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.model_metadata = pickle.load(f)
            
            # Load models
            for model_name in self.model_metadata.keys():
                if model_name == 'anomaly':
                    continue
                    
                model_path = self.models_dir / f"{model_name}_model.joblib"
                scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                
                if model_path.exists() and scaler_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
            
            if self.models:
                self.is_trained = True
                logger.info(f"Loaded {len(self.models)} trained models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models."""
        return {
            'is_trained': self.is_trained,
            'model_count': len(self.models),
            'training_samples': len(self.historical_insights),
            'models_available': list(self.models.keys()),
            'model_metadata': self.model_metadata,
            'models_dir': str(self.models_dir)
        }


# Factory function
def create_predictor(models_dir: Optional[Path] = None) -> PredictiveAnalytics:
    """
    Create a configured PredictiveAnalytics instance.
    
    Args:
        models_dir: Directory for model storage
        
    Returns:
        Configured predictor instance
    """
    return PredictiveAnalytics(models_dir=models_dir)