"""
Example Model Evaluator Implementations
Implement the ModelEvaluator protocol for your specific evaluation needs.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from pathlib import Path
import logging
import json


class BaseEvaluator:
    """Base class with common evaluation utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path("artifacts/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_plot(self, fig, filename: str, model_name: str) -> str:
        """Save matplotlib figure to file"""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        filepath = model_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _save_report(self, report: str, filename: str, model_name: str) -> str:
        """Save text report to file"""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        filepath = model_dir / f"{filename}.txt"
        with open(filepath, 'w') as f:
            f.write(report)
        
        return str(filepath)


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification models"""
    
    def evaluate(self, model: Any, test_data: pd.DataFrame, predictions: np.ndarray, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate classification model performance"""
        # Get true labels
        if 'target' in test_data.columns:
            y_true = test_data['target']
        else:
            raise ValueError("Target column 'target' not found in test data")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_true, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, predictions, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                # Get prediction probabilities
                X_test = test_data.drop(columns=['target'])
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        # Add per-class metrics for multiclass
        if len(np.unique(y_true)) > 2:
            per_class_f1 = f1_score(y_true, predictions, average=None)
            for i, f1 in enumerate(per_class_f1):
                metrics[f'f1_class_{i}'] = f1
        
        self.logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def generate_artifacts(self, model: Any, test_data: pd.DataFrame, predictions: np.ndarray, 
                          model_name: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate evaluation artifacts"""
        artifacts = {}
        
        # Get true labels and features
        y_true = test_data['target']
        X_test = test_data.drop(columns=['target'])
        
        try:
            # 1. Confusion Matrix
            artifacts['confusion_matrix'] = self._plot_confusion_matrix(y_true, predictions, model_name)
            
            # 2. Classification Report
            artifacts['classification_report'] = self._save_classification_report(y_true, predictions, model_name)
            
            # 3. ROC Curve (for binary classification)
            if len(np.unique(y_true)) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    artifacts['roc_curve'] = self._plot_roc_curve(y_true, y_proba, model_name)
                    artifacts['precision_recall_curve'] = self._plot_precision_recall_curve(y_true, y_proba, model_name)
                except:
                    self.logger.warning("Could not generate probability-based plots")
            
            # 4. Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                artifacts['feature_importance'] = self._plot_feature_importance(model, X_test.columns, model_name)
            elif hasattr(model, 'coef_'):
                artifacts['feature_coefficients'] = self._plot_feature_coefficients(model, X_test.columns, model_name)
            
            # 5. Prediction Distribution
            artifacts['prediction_distribution'] = self._plot_prediction_distribution(y_true, predictions, model_name)
            
        except Exception as e:
            self.logger.error(f"Error generating artifacts: {str(e)}")
        
        return artifacts
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str) -> str:
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        return self._save_plot(fig, 'confusion_matrix', model_name)
    
    def _save_classification_report(self, y_true, y_pred, model_name: str) -> str:
        """Save classification report"""
        report = classification_report(y_true, y_pred)
        return self._save_report(report, 'classification_report', model_name)
    
    def _plot_roc_curve(self, y_true, y_proba, model_name: str) -> str:
        """Plot ROC curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend()
        ax.grid(True)
        
        return self._save_plot(fig, 'roc_curve', model_name)
    
    def _plot_precision_recall_curve(self, y_true, y_proba, model_name: str) -> str:
        """Plot precision-recall curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.grid(True)
        
        return self._save_plot(fig, 'precision_recall_curve', model_name)
    
    def _plot_feature_importance(self, model, feature_names, model_name: str) -> str:
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        # Plot top features
        ax.bar(range(len(indices)), importances[indices])
        ax.set_title(f'Top 20 Feature Importances - {model_name}')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        
        # Set feature names on x-axis
        feature_labels = [feature_names[i] for i in indices]
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(feature_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        return self._save_plot(fig, 'feature_importance', model_name)
    
    def _plot_feature_coefficients(self, model, feature_names, model_name: str) -> str:
        """Plot feature coefficients for linear models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        indices = np.argsort(np.abs(coef))[::-1][:20]  # Top 20 by absolute value
        
        colors = ['red' if c < 0 else 'blue' for c in coef[indices]]
        ax.bar(range(len(indices)), coef[indices], color=colors)
        
        ax.set_title(f'Top 20 Feature Coefficients - {model_name}')
        ax.set_xlabel('Features')
        ax.set_ylabel('Coefficient Value')
        
        feature_labels = [feature_names[i] for i in indices]
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(feature_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        return self._save_plot(fig, 'feature_coefficients', model_name)
    
    def _plot_prediction_distribution(self, y_true, y_pred, model_name: str) -> str:
        """Plot prediction distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True labels distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar(unique_true, counts_true, alpha=0.7, color='blue')
        ax1.set_title('True Labels Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        
        # Predicted labels distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar(unique_pred, counts_pred, alpha=0.7, color='red')
        ax2.set_title('Predicted Labels Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        
        plt.suptitle(f'Label Distributions - {model_name}')
        plt.tight_layout()
        
        return self._save_plot(fig, 'prediction_distribution', model_name)


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression models"""
    
    def evaluate(self, model: Any, test_data: pd.DataFrame, predictions: np.ndarray, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate regression model performance"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Get true values
        if 'target' in test_data.columns:
            y_true = test_data['target']
        else:
            raise ValueError("Target column 'target' not found in test data")
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'mae': mean_absolute_error(y_true, predictions),
            'r2_score': r2_score(y_true, predictions)
        }
        
        # Add additional metrics
        metrics['mape'] = np.mean(np.abs((y_true - predictions) / y_true)) * 100  # Mean Absolute Percentage Error
        
        self.logger.info(f"Regression evaluation completed. R²: {metrics['r2_score']:.4f}")
        return metrics
    
    def generate_artifacts(self, model: Any, test_data: pd.DataFrame, predictions: np.ndarray, 
                          model_name: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate regression evaluation artifacts"""
        artifacts = {}
        
        y_true = test_data['target']
        X_test = test_data.drop(columns=['target'])
        
        try:
            # 1. Prediction vs True scatter plot
            artifacts['prediction_scatter'] = self._plot_prediction_scatter(y_true, predictions, model_name)
            
            # 2. Residuals plot
            artifacts['residuals_plot'] = self._plot_residuals(y_true, predictions, model_name)
            
            # 3. Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                artifacts['feature_importance'] = self._plot_feature_importance(model, X_test.columns, model_name)
            
            # 4. Error distribution
            artifacts['error_distribution'] = self._plot_error_distribution(y_true, predictions, model_name)
            
        except Exception as e:
            self.logger.error(f"Error generating regression artifacts: {str(e)}")
        
        return artifacts
    
    def _plot_prediction_scatter(self, y_true, y_pred, model_name: str) -> str:
        """Plot predicted vs true values"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predicted vs True Values - {model_name}')
        ax.legend()
        ax.grid(True)
        
        return self._save_plot(fig, 'prediction_scatter', model_name)
    
    def _plot_residuals(self, y_true, y_pred, model_name: str) -> str:
        """Plot residuals"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals Plot - {model_name}')
        ax.grid(True)
        
        return self._save_plot(fig, 'residuals_plot', model_name)
    
    def _plot_error_distribution(self, y_true, y_pred, model_name: str) -> str:
        """Plot error distribution"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        errors = y_true - y_pred
        ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution - {model_name}')
        ax.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        ax.legend()
        ax.grid(True)
        
        return self._save_plot(fig, 'error_distribution', model_name) 