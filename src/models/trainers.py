"""
Example Model Trainer Implementations
Implement the ModelTrainer protocol for your specific models.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import joblib
from pathlib import Path


class BaseTrainer:
    """Base class with common training utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate standard classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                pass
        
        return metrics
    
    def save_model(self, model, model_path: str) -> str:
        """Save model to disk"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return model_path


class RandomForestTrainer(BaseTrainer):
    """Trainer for Random Forest models"""
    
    def train(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train Random Forest model"""
        self.logger.info("Training Random Forest model...")
        
        # Separate features and target
        X = train_data.drop(columns=['target'])
        y = train_data['target']
        
        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})
        
        # Create and train model
        self.model = RandomForestClassifier(**hyperparams)
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Add model-specific metrics
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            # Log top 5 most important features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (feature, importance) in enumerate(top_features):
                metrics[f'top_feature_{i+1}_importance'] = importance
        
        metrics['oob_score'] = getattr(self.model, 'oob_score_', 0.0)
        
        self.logger.info(f"Random Forest training completed. Accuracy: {metrics['accuracy']:.4f}")
        return self.model, metrics
    
    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X = data.drop(columns=['target'], errors='ignore')
        return model.predict(X)


class GradientBoostTrainer(BaseTrainer):
    """Trainer for Gradient Boosting models"""
    
    def train(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train Gradient Boosting model"""
        self.logger.info("Training Gradient Boosting model...")
        
        # Separate features and target
        X = train_data.drop(columns=['target'])
        y = train_data['target']
        
        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})
        
        # Create and train model
        self.model = GradientBoostingClassifier(**hyperparams)
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Add model-specific metrics
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (feature, importance) in enumerate(top_features):
                metrics[f'top_feature_{i+1}_importance'] = importance
        
        metrics['train_score'] = getattr(self.model, 'train_score_', [])[-1] if hasattr(self.model, 'train_score_') else 0.0
        
        self.logger.info(f"Gradient Boosting training completed. Accuracy: {metrics['accuracy']:.4f}")
        return self.model, metrics
    
    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X = data.drop(columns=['target'], errors='ignore')
        return model.predict(X)


class LogisticRegressionTrainer(BaseTrainer):
    """Trainer for Logistic Regression models"""
    
    def train(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train Logistic Regression model"""
        self.logger.info("Training Logistic Regression model...")
        
        # Separate features and target
        X = train_data.drop(columns=['target'])
        y = train_data['target']
        
        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})
        
        # Create and train model
        self.model = LogisticRegression(**hyperparams)
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Add model-specific metrics
        if hasattr(self.model, 'coef_'):
            # Log coefficient statistics
            coef_stats = {
                'max_coefficient': float(np.max(np.abs(self.model.coef_))),
                'mean_coefficient': float(np.mean(np.abs(self.model.coef_))),
                'num_features': len(self.model.coef_[0])
            }
            metrics.update(coef_stats)
        
        self.logger.info(f"Logistic Regression training completed. Accuracy: {metrics['accuracy']:.4f}")
        return self.model, metrics
    
    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X = data.drop(columns=['target'], errors='ignore')
        return model.predict(X)


class SVMTrainer(BaseTrainer):
    """Trainer for Support Vector Machine models"""
    
    def train(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train SVM model"""
        self.logger.info("Training SVM model...")
        
        # Separate features and target
        X = train_data.drop(columns=['target'])
        y = train_data['target']
        
        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})
        
        # Create and train model
        self.model = SVC(probability=True, **hyperparams)  # Enable probability for consistent interface
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Add model-specific metrics
        if hasattr(self.model, 'support_'):
            metrics['num_support_vectors'] = len(self.model.support_)
            metrics['support_vector_ratio'] = len(self.model.support_) / len(X)
        
        self.logger.info(f"SVM training completed. Accuracy: {metrics['accuracy']:.4f}")
        return self.model, metrics
    
    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X = data.drop(columns=['target'], errors='ignore')
        return model.predict(X)


class NeuralNetworkTrainer(BaseTrainer):
    """Trainer for Neural Network models using sklearn MLPClassifier"""
    
    def train(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train Neural Network model"""
        from sklearn.neural_network import MLPClassifier
        
        self.logger.info("Training Neural Network model...")
        
        # Separate features and target
        X = train_data.drop(columns=['target'])
        y = train_data['target']
        
        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})
        
        # Convert hidden_layers list to tuple (required by MLPClassifier)
        if 'hidden_layers' in hyperparams:
            hyperparams['hidden_layer_sizes'] = tuple(hyperparams.pop('hidden_layers'))
        
        # Set default parameters for better convergence
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 1000,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        default_params.update(hyperparams)
        
        # Create and train model
        self.model = MLPClassifier(**default_params)
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Add model-specific metrics
        metrics.update({
            'training_loss': float(self.model.loss_),
            'num_iterations': int(self.model.n_iter_),
            'num_layers': len(self.model.coefs_),
            'converged': bool(self.model.n_iter_ < default_params.get('max_iter', 1000))
        })
        
        self.logger.info(f"Neural Network training completed. Accuracy: {metrics['accuracy']:.4f}")
        return self.model, metrics
    
    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X = data.drop(columns=['target'], errors='ignore')
        return model.predict(X)


class EnsembleTrainer(BaseTrainer):
    """Trainer for ensemble models combining multiple base models"""
    
    def __init__(self):
        super().__init__()
        self.base_models = []
    
    def train(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train ensemble model"""
        from sklearn.ensemble import VotingClassifier
        
        self.logger.info("Training Ensemble model...")
        
        # Separate features and target
        X = train_data.drop(columns=['target'])
        y = train_data['target']
        
        # Get base models configuration
        base_models_config = config.get('base_models', {
            'rf': {'class': RandomForestClassifier, 'params': {'n_estimators': 50}},
            'gb': {'class': GradientBoostingClassifier, 'params': {'n_estimators': 50}},
            'lr': {'class': LogisticRegression, 'params': {'max_iter': 1000}}
        })
        
        # Create base models
        estimators = []
        for name, model_config in base_models_config.items():
            model_class = model_config['class']
            model_params = model_config.get('params', {})
            model = model_class(**model_params)
            estimators.append((name, model))
        
        # Create ensemble
        voting_type = config.get('voting', 'hard')  # 'hard' or 'soft'
        self.model = VotingClassifier(estimators=estimators, voting=voting_type)
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        
        # For soft voting, we can get probabilities
        if voting_type == 'soft':
            y_proba = self.model.predict_proba(X)
        else:
            y_proba = None
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Add ensemble-specific metrics
        metrics['num_base_models'] = len(estimators)
        metrics['voting_type'] = voting_type
        
        self.logger.info(f"Ensemble training completed. Accuracy: {metrics['accuracy']:.4f}")
        return self.model, metrics
    
    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X = data.drop(columns=['target'], errors='ignore')
        return model.predict(X) 