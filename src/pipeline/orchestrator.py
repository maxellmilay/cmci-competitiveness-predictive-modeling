"""
ML Pipeline Orchestrator for CMCI Competitiveness Prediction
"""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import mlflow
import mlflow.sklearn
from datetime import datetime

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
        """Setup MLflow tracking"""
        mlflow.set_experiment("cmci_competitiveness_prediction")
    
    def run_data_pipeline(self) -> Dict[str, Any]:
        """Execute data processing pipeline"""
        self.logger.info("Starting data pipeline...")
        
        with mlflow.start_run(run_name="data_processing", nested=True):
            # Data ingestion
            self.logger.info("Step 1: Data ingestion")
            raw_data = ingestion.load_data(self.config['data']['raw_data_path'])
            mlflow.log_param("raw_data_shape", raw_data.shape)
            
            # Data cleaning
            self.logger.info("Step 2: Data cleaning")
            cleaned_data = cleaning.clean_data(
                raw_data, 
                strategy=self.config['preprocessing']['missing_value_strategy']
            )
            mlflow.log_param("cleaned_data_shape", cleaned_data.shape)
            
            # Feature engineering
            self.logger.info("Step 3: Feature engineering")
            features = build_features.engineer_features(
                cleaned_data,
                pca_components=self.config['feature_engineering']['pca_components']
            )
            mlflow.log_param("features_shape", features.shape)
            
            # Data splitting
            self.logger.info("Step 4: Data splitting")
            train_data, test_data = splitting.split_data(
                features,
                test_size=self.config['validation']['test_size']
            )
            
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
        """Execute model training pipeline"""
        self.logger.info("Starting model pipeline...")
        
        models = {}
        
        # Train Random Forest
        with mlflow.start_run(run_name="random_forest", nested=True):
            self.logger.info("Training Random Forest model")
            rf_model, rf_metrics = rf_train.train_model(
                data['train_data'],
                **self.config['model_training']['random_forest']
            )
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
            mlflow.log_metrics(rf_metrics)
            models['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
        
        # Train Gradient Boosting
        with mlflow.start_run(run_name="gradient_boost", nested=True):
            self.logger.info("Training Gradient Boosting model")
            gb_model, gb_metrics = gb_train.train_model(
                data['train_data'],
                **self.config['model_training']['gradient_boost']
            )
            mlflow.sklearn.log_model(gb_model, "gradient_boost_model")
            mlflow.log_metrics(gb_metrics)
            models['gradient_boost'] = {'model': gb_model, 'metrics': gb_metrics}
        
        return models
    
    def run_evaluation_pipeline(self, models: Dict[str, Any], test_data: Any) -> Dict[str, Any]:
        """Execute model evaluation pipeline"""
        self.logger.info("Starting evaluation pipeline...")
        
        evaluation_results = {}
        
        for model_name, model_info in models.items():
            with mlflow.start_run(run_name=f"evaluation_{model_name}", nested=True):
                self.logger.info(f"Evaluating {model_name}")
                
                # Generate predictions
                predictions = model_info['model'].predict(test_data)
                
                # Evaluate model
                metrics = evaluation.evaluate_model(test_data, predictions)
                mlflow.log_metrics(metrics)
                
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
        """Execute the complete ML pipeline"""
        self.logger.info("Starting full ML pipeline...")
        
        with mlflow.start_run(run_name="full_pipeline"):
            # Data pipeline
            data_results = self.run_data_pipeline()
            
            # Model pipeline
            models = self.run_model_pipeline(data_results)
            
            # Evaluation pipeline
            evaluation_results = self.run_evaluation_pipeline(
                models, 
                data_results['test_data']
            )
            
            # Final results
            results = {
                'data_results': data_results,
                'models': models,
                'evaluation_results': evaluation_results
            }
            
            self.logger.info("Full pipeline completed successfully!")
            return results


if __name__ == "__main__":
    # Example usage
    pipeline = CMCIPipeline()
    results = pipeline.run_full_pipeline()
    print("Pipeline completed successfully!")
