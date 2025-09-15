"""
User taste modeling system using machine learning to predict user preferences.
Supports both baseline (Logistic Regression) and advanced (LightGBM) models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None
import yaml

logger = logging.getLogger(__name__)


class TasteModel:
    """User taste modeling with multiple algorithm support."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize taste model with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_config = self.config['models']
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_type = None
        self.feature_importance = {}
        self.training_metrics = {}
        
    def prepare_training_data(self, df: pd.DataFrame, 
                            feature_columns: List[str],
                            target_column: str = 'liked',
                            balance_data: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        logger.info("Preparing training data")
        
        # Filter to rows with known preferences
        df_train = df[df[target_column].notna()].copy()
        
        if df_train.empty:
            raise ValueError("No training data available with target labels")
        
        # Select features
        available_features = [col for col in feature_columns if col in df_train.columns]
        self.feature_columns = available_features
        
        X = df_train[available_features].fillna(0)
        y = df_train[target_column]
        
        # Balance data if requested
        if balance_data:
            X, y = self._balance_dataset(X, y)
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X.values, y.values
    
    def _balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance dataset using undersampling/oversampling."""
        from sklearn.utils import resample
        
        # Combine X and y for resampling
        df_combined = X.copy()
        df_combined['target'] = y
        
        # Separate classes
        df_majority = df_combined[df_combined.target == 0]
        df_minority = df_combined[df_combined.target == 1]
        
        # Determine resampling strategy
        majority_size = len(df_majority)
        minority_size = len(df_minority)
        
        if majority_size > 2 * minority_size:
            # Undersample majority class
            df_majority_resampled = resample(
                df_majority,
                replace=False,
                n_samples=min(majority_size, 2 * minority_size),
                random_state=42
            )
            df_balanced = pd.concat([df_majority_resampled, df_minority])
        elif minority_size > 2 * majority_size:
            # Oversample minority class
            df_minority_resampled = resample(
                df_minority,
                replace=True,
                n_samples=min(minority_size, 2 * majority_size),
                random_state=42
            )
            df_balanced = pd.concat([df_majority, df_minority_resampled])
        else:
            # Classes are reasonably balanced
            df_balanced = df_combined
        
        # Shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X_balanced = df_balanced.drop('target', axis=1)
        y_balanced = df_balanced['target']
        
        logger.info(f"Balanced dataset: {len(X_balanced)} samples")
        return X_balanced, y_balanced
    
    def train_baseline_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train baseline logistic regression model."""
        logger.info("Training baseline logistic regression model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model_params = self.model_config['baseline_model']['params']
        self.model = LogisticRegression(**model_params)
        self.model.fit(X_train_scaled, y_train)
        self.model_type = 'logistic_regression'
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance (coefficients)
        self.feature_importance = dict(zip(
            self.feature_columns,
            np.abs(self.model.coef_[0])
        ))
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        self.training_metrics = metrics
        logger.info(f"Baseline model trained. AUC: {metrics['auc']:.3f}")
        
        return metrics
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train advanced LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available. Falling back to Random Forest.")
            return self.train_random_forest_model(X, y)
        
        logger.info("Training LightGBM model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Model parameters
        model_params = self.model_config['taste_model']['params'].copy()
        
        # Train model
        self.model = lgb.train(
            model_params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        self.model_type = 'lightgbm'
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Evaluate
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        importance_gain = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_columns, importance_gain))
        
        self.training_metrics = metrics
        logger.info(f"LightGBM model trained. AUC: {metrics['auc']:.3f}")
        
        return metrics
    
    def train_random_forest_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Random Forest model as an alternative."""
        logger.info("Training Random Forest model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.model_type = 'random_forest'
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        self.training_metrics = metrics
        logger.info(f"Random Forest model trained. AUC: {metrics['auc']:.3f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            model_type: str = 'lightgbm') -> Dict:
        """Perform hyperparameter tuning."""
        logger.info(f"Performing hyperparameter tuning for {model_type}")
        
        if model_type == 'lightgbm':
            return self._tune_lightgbm(X, y)
        elif model_type == 'logistic_regression':
            return self._tune_logistic_regression(X, y)
        elif model_type == 'random_forest':
            return self._tune_random_forest(X, y)
        else:
            logger.error(f"Unsupported model type for tuning: {model_type}")
            return {}
    
    def _tune_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune LightGBM hyperparameters."""
        from optuna import create_study, Trial
        
        def objective(trial: Trial) -> float:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1
            }
            
            # Cross-validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )
            
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            return roc_auc_score(y_val, y_pred)
        
        study = create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _tune_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune Logistic Regression hyperparameters."""
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga']
        }
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def _tune_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune Random Forest hyperparameters."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid,
            cv=3,  # Reduced CV folds due to computational cost
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def predict_preferences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict user preferences for new tracks."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_* method first.")
        
        logger.info(f"Predicting preferences for {len(df)} tracks")
        
        # Prepare features
        available_features = [col for col in self.feature_columns if col in df.columns]
        X = df[available_features].fillna(0)
        
        # Make predictions
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            preferences = self.model.predict(X.values, num_iteration=self.model.best_iteration)
        elif self.model_type in ['logistic_regression']:
            X_scaled = self.scaler.transform(X.values)
            preferences = self.model.predict_proba(X_scaled)[:, 1]
        else:  # random_forest or fallback
            preferences = self.model.predict_proba(X.values)[:, 1]
        
        # Add predictions to dataframe
        df_pred = df.copy()
        df_pred['preference_score'] = preferences
        df_pred['predicted_like'] = (preferences > 0.5).astype(int)
        
        return df_pred
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get top feature importance scores."""
        if not self.feature_importance:
            logger.warning("Feature importance not available")
            return {}
        
        # Sort by importance
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'top_features': sorted_importance[:top_n],
            'all_features': self.feature_importance,
            'feature_count': len(self.feature_importance)
        }
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None) -> bool:
        """Save trained model and metadata."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type,
                'feature_importance': self.feature_importance,
                'training_metrics': self.training_metrics,
                'metadata': metadata or {}
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model and metadata."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_metrics = model_data.get('training_metrics', {})
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Model type: {self.model_type}")
            logger.info(f"Features: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary."""
        if self.model is None:
            return {'status': 'not_trained'}
        
        summary = {
            'status': 'trained',
            'model_type': self.model_type,
            'feature_count': len(self.feature_columns),
            'training_metrics': self.training_metrics,
            'top_features': self.get_feature_importance(10)['top_features']
        }
        
        return summary
