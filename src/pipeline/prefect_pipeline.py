"""
Modern ML Pipeline using Prefect for CMCI Competitiveness Prediction
"""
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from prefect.logging import get_run_logger
import mlflow
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Import your existing modules
from src.data import ingestion, cleaning, build_features, splitting, validation
from src.models.random_forest import train as rf_train
from src.models.gradient_boost import train as gb_train
from src.visualization import evaluation


@task(name="load_config")
def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@task(name="ingest_data", retries=2)
def ingest_data_task(raw_data_path: str) -> pd.DataFrame:
    """Data ingestion task"""
    logger = get_run_logger()
    logger.info(f"Loading data from {raw_data_path}")
    return ingestion.load_data(raw_data_path)


@task(name="clean_data")
def clean_data_task(raw_data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Data cleaning task"""
    logger = get_run_logger()
    logger.info(f"Cleaning data with strategy: {strategy}")
    return cleaning.clean_data(raw_data, strategy=strategy)


@task(name="engineer_features")
def engineer_features_task(cleaned_data: pd.DataFrame, pca_components: int) -> pd.DataFrame:
    """Feature engineering task"""
    logger = get_run_logger()
    logger.info(f"Engineering features with {pca_components} PCA components")
    return build_features.engineer_features(cleaned_data, pca_components=pca_components)


@task(name="split_data")
def split_data_task(features: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Data splitting task"""
    logger = get_run_logger()
    logger.info(f"Splitting data with test_size: {test_size}")
    return splitting.split_data(features, test_size=test_size)


@task(name="validate_data")
def validate_data_task(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, float]:
    """Data validation task"""
    logger = get_run_logger()
    logger.info("Validating data splits")
    return validation.validate_data(train_data, test_data)


@task(name="train_random_forest")
def train_random_forest_task(train_data: pd.DataFrame, **model_params) -> Tuple[Any, Dict[str, float]]:
    """Random Forest training task"""
    logger = get_run_logger()
    logger.info("Training Random Forest model")
    return rf_train.train_model(train_data, **model_params)


@task(name="train_gradient_boost")
def train_gradient_boost_task(train_data: pd.DataFrame, **model_params) -> Tuple[Any, Dict[str, float]]:
    """Gradient Boosting training task"""
    logger = get_run_logger()
    logger.info("Training Gradient Boosting model")
    return gb_train.train_model(train_data, **model_params)


@task(name="evaluate_model")
def evaluate_model_task(model: Any, test_data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """Model evaluation task"""
    logger = get_run_logger()
    logger.info(f"Evaluating {model_name} model")
    
    predictions = model.predict(test_data)
    metrics = evaluation.evaluate_model(test_data, predictions)
    plots = evaluation.generate_plots(test_data, predictions, model_name)
    
    return {
        'predictions': predictions,
        'metrics': metrics,
        'plots': plots
    }


@flow(name="data_processing_flow", task_runner=ConcurrentTaskRunner())
def data_processing_flow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Data processing sub-flow"""
    logger = get_run_logger()
    logger.info("Starting data processing flow")
    
    # Execute data tasks
    raw_data = ingest_data_task(config['data']['raw_data_path'])
    cleaned_data = clean_data_task(raw_data, config['preprocessing']['missing_value_strategy'])
    features = engineer_features_task(cleaned_data, config['feature_engineering']['pca_components'])
    train_data, test_data = split_data_task(features, config['validation']['test_size'])
    validation_results = validate_data_task(train_data, test_data)
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'validation_results': validation_results
    }


@flow(name="model_training_flow", task_runner=ConcurrentTaskRunner())
def model_training_flow(config: Dict[str, Any], train_data: pd.DataFrame) -> Dict[str, Any]:
    """Model training sub-flow"""
    logger = get_run_logger()
    logger.info("Starting model training flow")
    
    # Train models concurrently
    rf_future = train_random_forest_task.submit(train_data, **config['model_training']['random_forest'])
    gb_future = train_gradient_boost_task.submit(train_data, **config['model_training']['gradient_boost'])
    
    # Wait for results
    rf_model, rf_metrics = rf_future.result()
    gb_model, gb_metrics = gb_future.result()
    
    return {
        'random_forest': {'model': rf_model, 'metrics': rf_metrics},
        'gradient_boost': {'model': gb_model, 'metrics': gb_metrics}
    }


@flow(name="evaluation_flow", task_runner=ConcurrentTaskRunner())
def evaluation_flow(models: Dict[str, Any], test_data: pd.DataFrame) -> Dict[str, Any]:
    """Model evaluation sub-flow"""
    logger = get_run_logger()
    logger.info("Starting evaluation flow")
    
    # Evaluate models concurrently
    evaluation_futures = {}
    for model_name, model_info in models.items():
        future = evaluate_model_task.submit(
            model_info['model'], 
            test_data, 
            model_name
        )
        evaluation_futures[model_name] = future
    
    # Collect results
    evaluation_results = {}
    for model_name, future in evaluation_futures.items():
        evaluation_results[model_name] = future.result()
    
    return evaluation_results


@flow(name="cmci_ml_pipeline", task_runner=ConcurrentTaskRunner())
def cmci_ml_pipeline(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Main CMCI ML Pipeline Flow"""
    logger = get_run_logger()
    logger.info("Starting CMCI ML Pipeline")
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup MLflow
    mlflow.set_experiment("cmci_competitiveness_prediction")
    
    with mlflow.start_run(run_name="prefect_pipeline"):
        # Execute sub-flows
        data_results = data_processing_flow(config)
        models = model_training_flow(config, data_results['train_data'])
        evaluation_results = evaluation_flow(models, data_results['test_data'])
        
        # Log artifacts to MLflow
        for model_name, model_info in models.items():
            with mlflow.start_run(run_name=f"mlflow_{model_name}", nested=True):
                mlflow.sklearn.log_model(model_info['model'], f"{model_name}_model")
                mlflow.log_metrics(model_info['metrics'])
                mlflow.log_metrics(evaluation_results[model_name]['metrics'])
        
        # Final results
        results = {
            'data_results': data_results,
            'models': models,
            'evaluation_results': evaluation_results
        }
        
        logger.info("CMCI ML Pipeline completed successfully!")
        return results


if __name__ == "__main__":
    # Run the pipeline
    results = cmci_ml_pipeline()
    print("Pipeline completed successfully!") 