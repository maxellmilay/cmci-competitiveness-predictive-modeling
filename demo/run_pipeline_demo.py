#!/usr/bin/env python3
"""
Demo Script: Generic ML Pipeline with MLflow
This script demonstrates the generic orchestrator with sample data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def create_sample_data():
    """Create sample dataset for demo"""
    print("🎯 Creating sample dataset...")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        class_sep=0.8
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['competitiveness_score'] = y  # This will be used to create target
    
    # Add some realistic business-like feature names
    business_features = [
        'market_share', 'revenue_growth', 'customer_satisfaction', 'innovation_index',
        'operational_efficiency', 'brand_strength', 'financial_stability', 'digital_adoption',
        'talent_quality', 'supply_chain_resilience', 'regulatory_compliance', 'sustainability_score',
        'customer_retention', 'product_quality', 'pricing_competitiveness', 'market_penetration',
        'technology_advancement', 'partnership_strength', 'geographical_reach', 'risk_management'
    ]
    
    df.columns = business_features + ['competitiveness_score']
    
    # Create data directory and save
    data_dir = Path("demo/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = data_dir / "sample_competitiveness_data.csv"
    df.to_csv(data_path, index=False)
    
    print(f"✅ Sample data created: {data_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {list(df.columns[:5])}... (and {len(df.columns)-5} more)")
    print(f"   Target distribution: {df['competitiveness_score'].value_counts().to_dict()}")
    
    return str(data_path)

def create_demo_config(data_path: str):
    """Create configuration for demo"""
    print("📋 Creating demo configuration...")
    
    config_content = f"""# Demo Configuration for Generic ML Pipeline
project_name: "competitiveness_demo"
experiment_name: "demo_run_v1"

# Data processing configuration
data_config:
  processor_class: "data.processors.CMCIDataProcessor"
  raw_data_path: "{data_path}"
  preprocessing:
    missing_value_strategy: "fillna"
    feature_selection: true
    normalize: true
  splitting:
    test_size: 0.2
    validation_size: 0.2
    random_state: 42
    stratify: true

# Models configuration
models_config:
  random_forest:
    trainer_class: "models.trainers.RandomForestTrainer"
    hyperparameters:
      n_estimators: 50  # Smaller for demo speed
      max_depth: 8
      random_state: 42
      n_jobs: -1
    
  logistic_regression:
    trainer_class: "models.trainers.LogisticRegressionTrainer"
    hyperparameters:
      C: 1.0
      max_iter: 1000
      random_state: 42
  
  gradient_boost:
    trainer_class: "models.trainers.GradientBoostTrainer"
    hyperparameters:
      n_estimators: 50  # Smaller for demo speed
      learning_rate: 0.1
      max_depth: 6
      random_state: 42

# Evaluation configuration
evaluation_config:
  evaluator_class: "evaluation.evaluators.ClassificationEvaluator"
  primary_metric: "f1_score"
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "roc_auc"
  generate_plots:
    - "confusion_matrix"
    - "roc_curve"
    - "feature_importance"
    - "learning_curves"

# MLflow configuration
mlflow_config:
  tracking_uri: "http://localhost:8080"
  auto_log: true
  log_models: true
  
# Logging configuration
logging_config:
  level: "INFO"
  log_dir: "demo/logs"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""
    
    # Create config directory and save
    config_dir = Path("demo/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / "demo_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Demo config created: {config_path}")
    return str(config_path)

def run_pipeline_demo():
    """Run the complete pipeline demo"""
    print("🚀 Starting Generic ML Pipeline Demo")
    print("=" * 50)
    
    try:
        # Step 1: Create sample data
        data_path = create_sample_data()
        
        # Step 2: Create demo configuration
        config_path = create_demo_config(data_path)
        
        # Step 3: Import and run pipeline
        print("🔄 Importing pipeline components...")
        from pipeline.generic_orchestrator import GenericMLPipeline
        
        print("🎯 Initializing pipeline...")
        pipeline = GenericMLPipeline(config_path)
        
        print("🚀 Running full pipeline...")
        print("   This will create experiments and runs in MLflow...")
        print("   Open http://localhost:8080 in your browser to see real-time results!")
        print()
        
        # Run the pipeline
        results = pipeline.run_full_pipeline()
        
        # Print results summary
        print()
        print("🎉 Pipeline completed successfully!")
        print("=" * 50)
        print(f"📊 Results Summary:")
        print(f"   Best Model: {results['best_model']}")
        print(f"   Models Trained: {list(results['trained_models'].keys())}")
        print()
        
        # Print model performance
        print("📈 Model Performance:")
        for model_name, eval_results in results['evaluation_results'].items():
            metrics = eval_results['metrics']
            print(f"   {model_name}:")
            print(f"     - Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"     - F1 Score: {metrics.get('f1_score', 0):.4f}")
            print(f"     - ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        
        print()
        print("🔗 MLflow UI Links:")
        print("   Main UI: http://localhost:8080")
        print("   Experiments: http://localhost:8080/#/experiments")
        print("   Models: http://localhost:8080/#/models")
        print()
        print("🎯 What to explore in MLflow UI:")
        print("   1. Experiment 'competitiveness_demo_demo_run_v1'")
        print("   2. Nested runs for each pipeline stage")
        print("   3. Model registry with registered models")
        print("   4. Artifacts (plots, reports, model files)")
        print("   5. Metrics comparison across models")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎯 Generic ML Pipeline Demo")
    print("This demo will:")
    print("1. Create sample competitiveness data")
    print("2. Configure a multi-model pipeline")
    print("3. Run the complete ML pipeline")
    print("4. Log everything to MLflow")
    print()
    print("Make sure MLflow server is running:")
    print("   mlflow server --host 0.0.0.0 --port 8080")
    print()
    
    input("Press Enter to continue...")
    results = run_pipeline_demo() 