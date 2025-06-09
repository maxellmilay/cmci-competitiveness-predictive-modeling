"""
ML Pipeline Orchestrator for CMCI Competitiveness Prediction
"""
import logging
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from datetime import datetime
import pandas as pd

# Import your existing modules
from src.data import ingestion, cleaning, build_features, splitting, validation
from src.models.random_forest import train as rf_train
from src.models.gradient_boost import train as gb_train
from src.visualization import evaluation, exploration


class CMCIPipeline:
    """Main pipeline orchestrator for CMCI competitiveness prediction"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        self.mlflow_client = MlflowClient()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking with proper configuration"""
        # Set tracking URI from environment or default
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
        mlflow.set_tracking_uri(uri=tracking_uri)
        
        # Set experiment
        experiment_name = "cmci_competitiveness_prediction"
        mlflow.set_experiment(experiment_name)
        
        self.logger.info(f"MLflow tracking URI: {tracking_uri}")
        self.logger.info(f"MLflow experiment: {experiment_name}")
    
    def _log_model_with_metadata(self, model, model_name: str, metrics: Dict[str, float], 
                                train_data: pd.DataFrame, predictions: Any) -> None:
        """Log model with comprehensive metadata and registration"""
        try:
            # Create model signature
            prediction_df = pd.DataFrame({
                "predictions": predictions,
                "competitiveness_score": predictions  # Adjust based on your target
            })
            signature = infer_signature(train_data.drop(columns=['target']), prediction_df)
            
            # Prepare input example
            input_example = train_data.drop(columns=['target']).head(3)
            
            # Prepare metadata
            metadata = {
                "model_type": model_name,
                "dataset": "cmci_competitiveness",
                "features": [str(col) for col in train_data.drop(columns=['target']).columns],
                "training_date": datetime.now().isoformat(),
                **{f"metric_{k}": float(v) for k, v in metrics.items()}
            }
            
            # Register model with comprehensive info
            registered_model_name = f"cmci-{model_name.lower().replace('_', '-')}-model"
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name}_model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                metadata=metadata
            )
            
            # Add model version tags for better tracking
            self._add_model_version_tags(registered_model_name, metrics, train_data, model_name)
            
            self.logger.info(f"Model {model_name} registered successfully: {model_info.model_uri}")
            
        except Exception as e:
            self.logger.error(f"Error logging model {model_name}: {str(e)}")
            # Fallback to basic logging
            mlflow.sklearn.log_model(model, f"{model_name}_model")
    
    def _add_model_version_tags(self, model_name: str, metrics: Dict[str, float], 
                               train_data: pd.DataFrame, model_type: str) -> None:
        """Add comprehensive tags to model version"""
        try:
            # Get latest version
            latest_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            if latest_versions:
                latest_version = max(latest_versions, key=lambda x: int(x.version))
                version = latest_version.version
                
                # Add metric tags
                for metric_name, metric_value in metrics.items():
                    self.mlflow_client.set_model_version_tag(
                        model_name, version, f"metric_{metric_name}", f"{metric_value:.4f}"
                    )
                
                # Add metadata tags
                feature_cols = [col for col in train_data.columns if col != 'target']
                self.mlflow_client.set_model_version_tag(model_name, version, "model_type", model_type)
                self.mlflow_client.set_model_version_tag(model_name, version, "dataset", "cmci_competitiveness")
                self.mlflow_client.set_model_version_tag(model_name, version, "num_features", str(len(feature_cols)))
                self.mlflow_client.set_model_version_tag(model_name, version, "training_samples", str(len(train_data)))
                self.mlflow_client.set_model_version_tag(model_name, version, "training_date", datetime.now().strftime("%Y-%m-%d"))
                
        except Exception as e:
            self.logger.warning(f"Could not add model version tags: {str(e)}")

    def run_data_pipeline(self) -> Dict[str, Any]:
        """Execute data processing pipeline"""
        self.logger.info("Starting data pipeline...")
        
        with mlflow.start_run(run_name="data_processing", nested=True):
            # Set pipeline tags
            mlflow.set_tag("pipeline_stage", "data_processing")
            mlflow.set_tag("pipeline_version", "1.0")
            
            # Data ingestion
            self.logger.info("Step 1: Data ingestion")
            raw_data = ingestion.load_data(self.config['data']['raw_data_path'])
            mlflow.log_param("raw_data_shape", raw_data.shape)
            mlflow.log_param("raw_data_path", self.config['data']['raw_data_path'])
            
            # Data cleaning
            self.logger.info("Step 2: Data cleaning")
            cleaned_data = cleaning.clean_data(
                raw_data, 
                strategy=self.config['preprocessing']['missing_value_strategy']
            )
            mlflow.log_param("cleaned_data_shape", cleaned_data.shape)
            mlflow.log_param("missing_value_strategy", self.config['preprocessing']['missing_value_strategy'])
            
            # Feature engineering
            self.logger.info("Step 3: Feature engineering")
            features = build_features.engineer_features(
                cleaned_data,
                pca_components=self.config['feature_engineering']['pca_components']
            )
            mlflow.log_param("features_shape", features.shape)
            mlflow.log_param("pca_components", self.config['feature_engineering']['pca_components'])
            
            # Data splitting
            self.logger.info("Step 4: Data splitting")
            train_data, test_data = splitting.split_data(
                features,
                test_size=self.config['validation']['test_size']
            )
            mlflow.log_param("test_size", self.config['validation']['test_size'])
            mlflow.log_param("train_size", len(train_data))
            mlflow.log_param("test_size_actual", len(test_data))
            
            # Data validation
            self.logger.info("Step 5: Data validation")
            validation_results = validation.validate_data(train_data, test_data)
            mlflow.log_metrics(validation_results)
            
            return {
                'train_data': train_data,
                'test_data': test_data,
                'validation_results': validation_results
            }
    
    def run_model_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training pipeline with enhanced MLflow tracking"""
        self.logger.info("Starting model pipeline...")
        
        models = {}
        train_data = data['train_data']
        
        # Train Random Forest
        with mlflow.start_run(run_name="random_forest_training", nested=True):
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("pipeline_stage", "training")
            
            self.logger.info("Training Random Forest model")
            rf_params = self.config['model_training']['random_forest']
            mlflow.log_params(rf_params)
            
            rf_model, rf_metrics = rf_train.train_model(train_data, **rf_params)
            mlflow.log_metrics(rf_metrics)
            
            # Generate predictions for signature
            rf_predictions = rf_model.predict(train_data.drop(columns=['target']))
            
            # Log model with comprehensive metadata
            self._log_model_with_metadata(rf_model, "random_forest", rf_metrics, train_data, rf_predictions)
            
            models['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
        
        # Train Gradient Boosting
        with mlflow.start_run(run_name="gradient_boost_training", nested=True):
            mlflow.set_tag("model_type", "GradientBoosting")
            mlflow.set_tag("pipeline_stage", "training")
            
            self.logger.info("Training Gradient Boosting model")
            gb_params = self.config['model_training']['gradient_boost']
            mlflow.log_params(gb_params)
            
            gb_model, gb_metrics = gb_train.train_model(train_data, **gb_params)
            mlflow.log_metrics(gb_metrics)
            
            # Generate predictions for signature
            gb_predictions = gb_model.predict(train_data.drop(columns=['target']))
            
            # Log model with comprehensive metadata
            self._log_model_with_metadata(gb_model, "gradient_boost", gb_metrics, train_data, gb_predictions)
            
            models['gradient_boost'] = {'model': gb_model, 'metrics': gb_metrics}
        
        return models
    
    def run_evaluation_pipeline(self, models: Dict[str, Any], test_data: Any) -> Dict[str, Any]:
        """Execute model evaluation pipeline"""
        self.logger.info("Starting evaluation pipeline...")
        
        evaluation_results = {}
        
        for model_name, model_info in models.items():
            with mlflow.start_run(run_name=f"evaluation_{model_name}", nested=True):
                mlflow.set_tag("pipeline_stage", "evaluation")
                mlflow.set_tag("model_type", model_name)
                
                self.logger.info(f"Evaluating {model_name}")
                
                # Generate predictions
                predictions = model_info['model'].predict(test_data.drop(columns=['target']))
                
                # Evaluate model
                metrics = evaluation.evaluate_model(test_data, predictions)
                mlflow.log_metrics(metrics)
                
                # Log evaluation metadata
                mlflow.log_param("test_samples", len(test_data))
                mlflow.log_param("model_name", model_name)
                
                # Generate visualizations
                plots = evaluation.generate_plots(test_data, predictions, model_name)
                for plot_name, plot_path in plots.items():
                    mlflow.log_artifact(plot_path)
                
                evaluation_results[model_name] = {
                    'predictions': predictions,
                    'metrics': metrics,
                    'plots': plots
                }
        
        return evaluation_results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute the complete ML pipeline with comprehensive MLflow tracking"""
        self.logger.info("Starting full ML pipeline...")
        
        with mlflow.start_run(run_name="cmci_full_pipeline"):
            # Set main pipeline tags
            mlflow.set_tag("pipeline_type", "full_cmci_pipeline")
            mlflow.set_tag("pipeline_version", "1.0")
            mlflow.set_tag("execution_date", datetime.now().isoformat())
            
            # Log pipeline configuration
            mlflow.log_params({
                "config_file": "pipeline_config.yaml",
                "python_version": "3.x",
                "mlflow_version": mlflow.__version__
            })
            
            # Data pipeline
            data_results = self.run_data_pipeline()
            
            # Model pipeline
            models = self.run_model_pipeline(data_results)
            
            # Evaluation pipeline
            evaluation_results = self.run_evaluation_pipeline(
                models, 
                data_results['test_data']
            )
            
            # Log final pipeline metrics
            best_model = max(models.keys(), key=lambda k: models[k]['metrics'].get('accuracy', 0))
            mlflow.log_param("best_model", best_model)
            mlflow.log_metric("best_model_accuracy", models[best_model]['metrics'].get('accuracy', 0))
            
            # Final results
            results = {
                'data_results': data_results,
                'models': models,
                'evaluation_results': evaluation_results,
                'best_model': best_model
            }
            
            self.logger.info("Full pipeline completed successfully!")
            self.logger.info(f"Best model: {best_model}")
            return results


if __name__ == "__main__":
    # Example usage
    pipeline = CMCIPipeline()
    results = pipeline.run_full_pipeline()
    print("Pipeline completed successfully!")
    print(f"Best model: {results['best_model']}")
