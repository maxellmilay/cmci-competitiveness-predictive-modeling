"""
Generic ML Pipeline Orchestrator with MLflow Best Practices
A reusable orchestrator for any ML project with proper experiment tracking.
"""
import logging
import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
# Optional imports for different model frameworks
try:
    import mlflow.pytorch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import mlflow.tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
from mlflow.tracking import MlflowClient
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import importlib
import traceback


@dataclass
class PipelineConfig:
    """Configuration class for pipeline settings"""
    project_name: str
    experiment_name: str
    data_config: Dict[str, Any]
    models_config: Dict[str, Any] 
    evaluation_config: Dict[str, Any]
    mlflow_config: Dict[str, Any]
    logging_config: Dict[str, Any]


class DataProcessor(Protocol):
    """Protocol for data processing components"""
    
    def load_data(self, config: Dict[str, Any]) -> Any:
        """Load raw data"""
        pass
    
    def preprocess(self, data: Any, config: Dict[str, Any]) -> Any:
        """Preprocess data"""
        pass
    
    def split_data(self, data: Any, config: Dict[str, Any]) -> tuple:
        """Split data into train/test"""
        pass


class ModelTrainer(Protocol):
    """Protocol for model training components"""
    
    def train(self, train_data: Any, config: Dict[str, Any]) -> tuple:
        """Train model and return (model, metrics)"""
        pass
    
    def predict(self, model: Any, data: Any) -> Any:
        """Generate predictions"""
        pass


class ModelEvaluator(Protocol):
    """Protocol for model evaluation components"""
    
    def evaluate(self, model: Any, test_data: Any, predictions: Any, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    def generate_artifacts(self, model: Any, test_data: Any, predictions: Any, model_name: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate evaluation artifacts (plots, reports)"""
        pass


class GenericMLPipeline:
    """
    Generic ML Pipeline Orchestrator with MLflow Best Practices
    
    This orchestrator can be reused across different ML projects by:
    1. Configuring data processors, models, and evaluators via config
    2. Following MLflow best practices for experiment tracking
    3. Providing modular, pluggable components
    """
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        self.config = self._load_and_validate_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        self.mlflow_client = MlflowClient()
        
        # Load pluggable components
        self.data_processor = self._load_component(self.config.data_config['processor_class'])
        self.models = self._load_models()
        self.evaluator = self._load_component(self.config.evaluation_config['evaluator_class'])
    
    def _load_and_validate_config(self, config_path: str) -> PipelineConfig:
        """Load and validate pipeline configuration"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['project_name', 'experiment_name', 'data_config', 'models_config', 'evaluation_config']
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required config section: {section}")
        
        # Set defaults
        config_dict.setdefault('mlflow_config', {})
        config_dict.setdefault('logging_config', {'level': 'INFO', 'log_dir': 'logs'})
        
        return PipelineConfig(**config_dict)
    
    def _setup_logging(self) -> None:
        """Setup logging with best practices"""
        log_config = self.config.logging_config
        log_dir = Path(log_config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Create structured log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{self.config.project_name}_{timestamp}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized pipeline for project: {self.config.project_name}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow with best practices"""
        mlflow_config = self.config.mlflow_config
        
        # Set tracking URI (environment > config > default)
        tracking_uri = (
            os.getenv("MLFLOW_TRACKING_URI") or 
            mlflow_config.get('tracking_uri') or 
            "http://127.0.0.1:5000"
        )
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment with proper naming
        experiment_name = f"{self.config.project_name}_{self.config.experiment_name}"
        mlflow.set_experiment(experiment_name)
        
        # Set additional MLflow configurations
        if 'registry_uri' in mlflow_config:
            mlflow.set_registry_uri(mlflow_config['registry_uri'])
        
        self.logger.info(f"MLflow tracking URI: {tracking_uri}")
        self.logger.info(f"MLflow experiment: {experiment_name}")
    
    def _load_component(self, component_path: str):
        """Dynamically load a component class"""
        try:
            module_path, class_name = component_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)()
        except Exception as e:
            self.logger.error(f"Failed to load component {component_path}: {str(e)}")
            raise
    
    def _load_models(self) -> Dict[str, ModelTrainer]:
        """Load all configured model trainers"""
        models = {}
        for model_name, model_config in self.config.models_config.items():
            try:
                model_trainer = self._load_component(model_config['trainer_class'])
                models[model_name] = model_trainer
                self.logger.info(f"Loaded model trainer: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {str(e)}")
        return models
    
    def _log_system_info(self) -> None:
        """Log system and environment information"""
        try:
            import platform
            import psutil
            
            system_info = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "mlflow_version": mlflow.__version__
            }
            mlflow.log_params(system_info)
        except Exception as e:
            self.logger.warning(f"Could not log system info: {str(e)}")
    
    def _create_model_signature(self, model: Any, train_data: Any, predictions: Any) -> Any:
        """Create MLflow model signature with error handling"""
        try:
            # Handle different data types
            if hasattr(train_data, 'drop'):  # DataFrame
                input_data = train_data.drop(columns=['target'], errors='ignore')
            else:
                input_data = train_data
            
            # Create prediction DataFrame for better schema
            if isinstance(predictions, np.ndarray):
                if predictions.ndim == 1:
                    pred_df = pd.DataFrame({"prediction": predictions})
                else:
                    pred_df = pd.DataFrame(predictions, columns=[f"output_{i}" for i in range(predictions.shape[1])])
            else:
                pred_df = pd.DataFrame({"prediction": predictions})
            
            return infer_signature(input_data, pred_df)
        except Exception as e:
            self.logger.warning(f"Could not create model signature: {str(e)}")
            return None
    
    def _log_model_with_registry(self, model: Any, model_name: str, metrics: Dict[str, float], 
                                train_data: Any, predictions: Any, model_config: Dict[str, Any]) -> None:
        """Log model to MLflow with registry and comprehensive metadata"""
        try:
            # Create signature
            signature = self._create_model_signature(model, train_data, predictions)
            
            # Prepare input example
            if hasattr(train_data, 'head'):
                input_example = train_data.drop(columns=['target'], errors='ignore').head(3)
            else:
                input_example = train_data[:3] if len(train_data) > 3 else train_data
            
            # Prepare comprehensive metadata
            metadata = {
                "model_type": model_name,
                "project": self.config.project_name,
                "experiment": self.config.experiment_name,
                "training_date": datetime.now().isoformat(),
                "config": json.dumps(model_config, default=str),
                **{f"metric_{k}": float(v) for k, v in metrics.items()}
            }
            
            # Determine model flavor based on model type
            model_flavor = self._detect_model_flavor(model)
            
            # Register model with proper naming convention
            registered_model_name = f"{self.config.project_name}-{model_name.lower().replace('_', '-')}"
            
            if model_flavor == "sklearn":
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{model_name}_model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    metadata=metadata
                )
            elif model_flavor == "pytorch" and HAS_PYTORCH:
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=f"{model_name}_model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    metadata=metadata
                )
            elif model_flavor == "tensorflow" and HAS_TENSORFLOW:
                model_info = mlflow.tensorflow.log_model(
                    tf_model=model,
                    artifact_path=f"{model_name}_model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    metadata=metadata
                )
            else:
                # Fallback to sklearn logging for most models
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{model_name}_model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    metadata=metadata
                )
            
            # Add model version tags
            self._add_model_version_tags(registered_model_name, metrics, model_config, model_name)
            
            self.logger.info(f"Model {model_name} registered: {model_info.model_uri}")
            
        except Exception as e:
            self.logger.error(f"Error logging model {model_name}: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Fallback to basic logging
            mlflow.log_artifact("fallback_model.pkl")
    
    def _detect_model_flavor(self, model: Any) -> str:
        """Detect appropriate MLflow model flavor"""
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            return "sklearn"
        elif hasattr(model, 'state_dict'):
            return "pytorch"
        elif hasattr(model, 'save'):
            return "tensorflow"
        else:
            return "generic"
    
    def _add_model_version_tags(self, model_name: str, metrics: Dict[str, float], 
                               model_config: Dict[str, Any], model_type: str) -> None:
        """Add comprehensive tags to model version for better tracking"""
        try:
            latest_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            if latest_versions:
                latest_version = max(latest_versions, key=lambda x: int(x.version))
                version = latest_version.version
                
                # Performance tags
                for metric_name, metric_value in metrics.items():
                    self.mlflow_client.set_model_version_tag(
                        model_name, version, f"perf_{metric_name}", f"{metric_value:.4f}"
                    )
                
                # Model metadata tags
                tags = {
                    "model_type": model_type,
                    "project": self.config.project_name,
                    "experiment": self.config.experiment_name,
                    "training_date": datetime.now().strftime("%Y-%m-%d"),
                    "status": "staging"  # Default status
                }
                
                # Add config-specific tags
                if 'hyperparameters' in model_config:
                    for param, value in model_config['hyperparameters'].items():
                        tags[f"param_{param}"] = str(value)
                
                # Set all tags
                for tag_key, tag_value in tags.items():
                    self.mlflow_client.set_model_version_tag(model_name, version, tag_key, tag_value)
                
        except Exception as e:
            self.logger.warning(f"Could not add model version tags: {str(e)}")
    
    def run_data_pipeline(self) -> Dict[str, Any]:
        """Execute data processing pipeline with comprehensive tracking"""
        self.logger.info("Starting data pipeline...")
        
        with mlflow.start_run(run_name="data_processing", nested=True):
            mlflow.set_tag("pipeline_stage", "data_processing")
            mlflow.set_tag("project", self.config.project_name)
            
            try:
                # Log data configuration
                mlflow.log_params(self.config.data_config)
                
                # Data loading
                self.logger.info("Loading data...")
                raw_data = self.data_processor.load_data(self.config.data_config)
                
                # Log data statistics
                if hasattr(raw_data, 'shape'):
                    mlflow.log_param("raw_data_shape", raw_data.shape)
                    mlflow.log_param("raw_data_size", raw_data.size if hasattr(raw_data, 'size') else len(raw_data))
                
                # Data preprocessing
                self.logger.info("Preprocessing data...")
                processed_data = self.data_processor.preprocess(raw_data, self.config.data_config)
                
                if hasattr(processed_data, 'shape'):
                    mlflow.log_param("processed_data_shape", processed_data.shape)
                
                # Data splitting
                self.logger.info("Splitting data...")
                train_data, test_data = self.data_processor.split_data(processed_data, self.config.data_config)
                
                # Log split information
                if hasattr(train_data, '__len__') and hasattr(test_data, '__len__'):
                    mlflow.log_param("train_size", len(train_data))
                    mlflow.log_param("test_size", len(test_data))
                    mlflow.log_param("train_test_ratio", len(train_data) / len(test_data))
                
                return {
                    'raw_data': raw_data,
                    'processed_data': processed_data,
                    'train_data': train_data,
                    'test_data': test_data
                }
                
            except Exception as e:
                self.logger.error(f"Data pipeline failed: {str(e)}")
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
                raise
    
    def run_model_training(self, train_data: Any) -> Dict[str, Any]:
        """Execute model training pipeline with parallel capability"""
        self.logger.info("Starting model training...")
        
        trained_models = {}
        
        for model_name, model_trainer in self.models.items():
            with mlflow.start_run(run_name=f"training_{model_name}", nested=True):
                mlflow.set_tag("pipeline_stage", "training")
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("project", self.config.project_name)
                
                try:
                    self.logger.info(f"Training {model_name}...")
                    
                    # Get model-specific config
                    model_config = self.config.models_config[model_name]
                    mlflow.log_params(model_config.get('hyperparameters', {}))
                    
                    # Train model
                    model, metrics = model_trainer.train(train_data, model_config)
                    
                    # Log training metrics
                    mlflow.log_metrics(metrics)
                    
                    # Generate predictions for signature
                    predictions = model_trainer.predict(model, train_data)
                    
                    # Log model with registry
                    self._log_model_with_registry(model, model_name, metrics, train_data, predictions, model_config)
                    
                    trained_models[model_name] = {
                        'model': model,
                        'trainer': model_trainer,
                        'metrics': metrics,
                        'config': model_config
                    }
                    
                    mlflow.set_tag("status", "success")
                    self.logger.info(f"Successfully trained {model_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {model_name}: {str(e)}")
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error", str(e))
                    # Continue with other models
        
        return trained_models
    
    def run_model_evaluation(self, trained_models: Dict[str, Any], test_data: Any) -> Dict[str, Any]:
        """Execute model evaluation pipeline"""
        self.logger.info("Starting model evaluation...")
        
        evaluation_results = {}
        
        for model_name, model_info in trained_models.items():
            with mlflow.start_run(run_name=f"evaluation_{model_name}", nested=True):
                mlflow.set_tag("pipeline_stage", "evaluation")
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("project", self.config.project_name)
                
                try:
                    self.logger.info(f"Evaluating {model_name}...")
                    
                    model = model_info['model']
                    trainer = model_info['trainer']
                    
                    # Generate predictions
                    predictions = trainer.predict(model, test_data)
                    
                    # Evaluate model
                    metrics = self.evaluator.evaluate(model, test_data, predictions, self.config.evaluation_config)
                    mlflow.log_metrics(metrics)
                    
                    # Generate artifacts
                    artifacts = self.evaluator.generate_artifacts(
                        model, test_data, predictions, model_name, self.config.evaluation_config
                    )
                    
                    # Log artifacts
                    for artifact_name, artifact_path in artifacts.items():
                        if os.path.exists(artifact_path):
                            mlflow.log_artifact(artifact_path, f"evaluation/{artifact_name}")
                    
                    evaluation_results[model_name] = {
                        'predictions': predictions,
                        'metrics': metrics,
                        'artifacts': artifacts
                    }
                    
                    mlflow.set_tag("status", "success")
                    
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error", str(e))
        
        return evaluation_results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute the complete ML pipeline with best practices"""
        self.logger.info(f"Starting full ML pipeline for {self.config.project_name}")
        
        with mlflow.start_run(run_name=f"{self.config.project_name}_full_pipeline"):
            # Set main pipeline tags and metadata
            mlflow.set_tag("pipeline_type", "full_ml_pipeline")
            mlflow.set_tag("project", self.config.project_name)
            mlflow.set_tag("experiment", self.config.experiment_name)
            mlflow.set_tag("execution_date", datetime.now().isoformat())
            
            # Log system and configuration info
            self._log_system_info()
            mlflow.log_params({
                "project_name": self.config.project_name,
                "num_models": len(self.config.models_config),
                "models": list(self.config.models_config.keys())
            })
            
            try:
                # Execute pipeline stages
                data_results = self.run_data_pipeline()
                trained_models = self.run_model_training(data_results['train_data'])
                evaluation_results = self.run_model_evaluation(trained_models, data_results['test_data'])
                
                # Determine best model
                best_model_name = self._select_best_model(evaluation_results)
                
                # Log final pipeline results
                if best_model_name:
                    mlflow.log_param("best_model", best_model_name)
                    best_metrics = evaluation_results[best_model_name]['metrics']
                    mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
                
                mlflow.set_tag("status", "success")
                
                results = {
                    'data_results': data_results,
                    'trained_models': trained_models,
                    'evaluation_results': evaluation_results,
                    'best_model': best_model_name,
                    'pipeline_status': 'success'
                }
                
                self.logger.info("Pipeline completed successfully!")
                if best_model_name:
                    self.logger.info(f"Best model: {best_model_name}")
                
                return results
                
            except Exception as e:
                self.logger.error(f"Pipeline failed: {str(e)}")
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
                mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
                raise
    
    def _select_best_model(self, evaluation_results: Dict[str, Any]) -> Optional[str]:
        """Select best model based on primary metric"""
        if not evaluation_results:
            return None
        
        # Get primary metric from config (default to accuracy)
        primary_metric = self.config.evaluation_config.get('primary_metric', 'accuracy')
        
        best_model = None
        best_score = float('-inf')
        
        for model_name, results in evaluation_results.items():
            metrics = results.get('metrics', {})
            if primary_metric in metrics:
                score = metrics[primary_metric]
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model


# Example usage and configuration templates
if __name__ == "__main__":
    # Example: Run pipeline with configuration
    pipeline = GenericMLPipeline("config/generic_pipeline_config.yaml")
    results = pipeline.run_full_pipeline()
    print(f"Pipeline completed! Best model: {results.get('best_model')}") 