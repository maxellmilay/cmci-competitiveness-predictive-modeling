# MLflow Best Practices Guide
## Building Reusable ML Pipelines with Proper Experiment Tracking

This guide demonstrates how to build production-ready ML pipelines with MLflow best practices using our generic orchestrator.

## 🎯 **Key MLflow Best Practices**

### 1. **Experiment Organization**
```python
# ✅ Good: Structured experiment naming
experiment_name = f"{project_name}_{version}_{environment}"
mlflow.set_experiment("cmci_competitiveness_production_v1")

# ❌ Bad: Generic names
mlflow.set_experiment("experiment1")
```

### 2. **Hierarchical Run Structure**
```python
# ✅ Good: Nested runs for pipeline stages
with mlflow.start_run(run_name="main_pipeline"):
    with mlflow.start_run(run_name="data_processing", nested=True):
        # Data processing code
    with mlflow.start_run(run_name="model_training", nested=True):
        # Model training code
```

### 3. **Comprehensive Model Registry**
```python
# ✅ Good: Rich model metadata
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature,              # Input/output schema
    input_example=input_example,      # Sample input
    registered_model_name="project-model-v1",
    metadata={
        "training_date": "2024-01-15",
        "features": feature_list,
        "performance": metrics
    }
)
```

### 4. **Model Versioning & Tagging**
```python
# ✅ Good: Semantic versioning and status tags
client.set_model_version_tag(model_name, version, "status", "staging")
client.set_model_version_tag(model_name, version, "performance", "0.95")
client.set_model_version_tag(model_name, version, "dataset_version", "v2.1")
```

## 🚀 **Using the Generic Orchestrator**

### **Step 1: Configure Your Pipeline**

Create a configuration file for your specific project:

```yaml
# config/my_project_config.yaml
project_name: "customer_churn"
experiment_name: "production_models_v2"

data_config:
  processor_class: "src.data.processors.TabularDataProcessor"
  raw_data_path: "data/customer_data.csv"
  preprocessing:
    missing_value_strategy: "fillna"
    normalize: true

models_config:
  random_forest:
    trainer_class: "src.models.trainers.RandomForestTrainer"
    hyperparameters:
      n_estimators: 200
      max_depth: 15
      random_state: 42
  
  logistic_regression:
    trainer_class: "src.models.trainers.LogisticRegressionTrainer"
    hyperparameters:
      C: 1.0
      max_iter: 1000

evaluation_config:
  evaluator_class: "src.evaluation.evaluators.ClassificationEvaluator"
  primary_metric: "f1_score"
```

### **Step 2: Run Your Pipeline**

```python
from src.pipeline.generic_orchestrator import GenericMLPipeline

# Initialize and run pipeline
pipeline = GenericMLPipeline("config/my_project_config.yaml")
results = pipeline.run_full_pipeline()

print(f"Best model: {results['best_model']}")
print(f"Performance: {results['evaluation_results'][results['best_model']]['metrics']}")
```

### **Step 3: Access Results in MLflow UI**

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Open browser to http://localhost:5000
```

## 📊 **What Gets Tracked**

### **Experiments & Runs**
- ✅ Hierarchical run structure (main → data → training → evaluation)
- ✅ Comprehensive parameter logging
- ✅ Metric tracking across all stages
- ✅ System information and environment details

### **Models**
- ✅ Automatic model registration with semantic names
- ✅ Input/output signatures for schema validation
- ✅ Comprehensive metadata and version tags
- ✅ Performance metrics and training information

### **Artifacts**
- ✅ Evaluation plots (confusion matrix, ROC curves)
- ✅ Feature importance/coefficients
- ✅ Classification reports
- ✅ Model files and preprocessing objects

### **Tags for Organization**
- ✅ Pipeline stage tags (`data_processing`, `training`, `evaluation`)
- ✅ Model type tags (`RandomForest`, `LogisticRegression`)
- ✅ Status tags (`staging`, `production`, `archived`)
- ✅ Performance tags for quick filtering

## 🔧 **Extending the Orchestrator**

### **Adding New Data Processors**

```python
class MyCustomProcessor(BaseDataProcessor):
    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        # Your custom data loading logic
        return data
    
    def preprocess(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        # Your custom preprocessing logic
        return processed_data
    
    def split_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Your custom splitting logic
        return train_data, test_data
```

### **Adding New Model Trainers**

```python
class MyCustomTrainer(BaseTrainer):
    def train(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        # Your custom training logic
        model = MyModel(**config['hyperparameters'])
        model.fit(train_data)
        
        metrics = self._calculate_metrics(y_true, y_pred)
        return model, metrics
    
    def predict(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        return model.predict(data)
```

### **Adding New Evaluators**

```python
class MyCustomEvaluator(BaseEvaluator):
    def evaluate(self, model: Any, test_data: pd.DataFrame, predictions: np.ndarray, config: Dict[str, Any]) -> Dict[str, float]:
        # Your custom evaluation metrics
        return metrics
    
    def generate_artifacts(self, model: Any, test_data: pd.DataFrame, predictions: np.ndarray, 
                          model_name: str, config: Dict[str, Any]) -> Dict[str, str]:
        # Your custom visualization and reporting
        return artifacts
```

## 🏗️ **Production Deployment Workflow**

### **1. Model Development**
```python
# Development experiment
config = load_config("config/dev_config.yaml")
pipeline = GenericMLPipeline(config)
results = pipeline.run_full_pipeline()
```

### **2. Model Validation**
```python
# Validation on holdout data
client = MlflowClient()
model_version = client.get_latest_versions("my-model", stages=["Staging"])[0]
client.set_model_version_tag(model_version.name, model_version.version, "validation_status", "passed")
```

### **3. Model Promotion**
```python
# Promote to production
client.transition_model_version_stage(
    name="my-model",
    version=model_version.version,
    stage="Production"
)
```

### **4. Model Serving**
```python
# Load production model
model = mlflow.sklearn.load_model(f"models:/my-model/Production")
predictions = model.predict(new_data)
```

## 🎯 **Real-World Examples**

### **Example 1: Different Projects**

```yaml
# E-commerce Recommendation
project_name: "ecommerce_recommendations"
data_config:
  processor_class: "src.data.processors.RecommendationDataProcessor"
models_config:
  collaborative_filtering:
    trainer_class: "src.models.trainers.CollaborativeFilteringTrainer"

# Fraud Detection
project_name: "fraud_detection"
data_config:
  processor_class: "src.data.processors.FraudDataProcessor"
models_config:
  isolation_forest:
    trainer_class: "src.models.trainers.IsolationForestTrainer"
```

### **Example 2: A/B Testing Models**

```python
# Experiment A: Traditional approach
pipeline_a = GenericMLPipeline("config/experiment_a.yaml")
results_a = pipeline_a.run_full_pipeline()

# Experiment B: Deep learning approach
pipeline_b = GenericMLPipeline("config/experiment_b.yaml")
results_b = pipeline_b.run_full_pipeline()

# Compare in MLflow UI
```

## 📈 **Monitoring & Maintenance**

### **Performance Tracking**
```python
# Log production metrics
with mlflow.start_run(run_name="production_monitoring"):
    mlflow.log_metric("accuracy", current_accuracy)
    mlflow.log_metric("data_drift", drift_score)
    mlflow.log_metric("prediction_latency", latency_ms)
```

### **Model Retraining Triggers**
```python
# Automated retraining based on performance degradation
if current_accuracy < threshold:
    pipeline = GenericMLPipeline("config/retrain_config.yaml")
    results = pipeline.run_full_pipeline()
```

## 🎓 **Learning Path**

1. **Start Simple**: Use the basic configuration with existing components
2. **Customize Gradually**: Add your own data processors and model trainers
3. **Scale Up**: Implement parallel training and advanced evaluation
4. **Production Ready**: Add monitoring, automated retraining, and deployment

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Model Registration Fails**
   ```python
   # Check MLflow tracking URI
   print(mlflow.get_tracking_uri())
   
   # Ensure model registry is accessible
   client = MlflowClient()
   print(client.list_registered_models())
   ```

2. **Signature Inference Fails**
   ```python
   # Manual signature creation
   from mlflow.types.schema import Schema, ColSpec
   input_schema = Schema([ColSpec("double", "feature1"), ColSpec("double", "feature2")])
   output_schema = Schema([ColSpec("long", "prediction")])
   signature = ModelSignature(inputs=input_schema, outputs=output_schema)
   ```

3. **Configuration Errors**
   ```python
   # Validate configuration before running
   pipeline = GenericMLPipeline("config/my_config.yaml")
   # Check if all components load successfully
   ```

## 🚀 **Next Steps**

1. **Try the Examples**: Start with the provided configurations
2. **Experiment**: Modify configurations for your specific use case
3. **Extend**: Add custom components as needed
4. **Scale**: Deploy to production with proper monitoring

This orchestrator provides a solid foundation for building reproducible, trackable ML pipelines while following MLflow best practices! 