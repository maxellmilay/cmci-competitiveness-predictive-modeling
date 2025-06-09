import mlflow
from mlflow.tracking import MlflowClient
import json

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Initialize the MLflow client
client = MlflowClient()

# Get the latest version of the model
model_name = "iris-classification-model"
latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

print(f"🔍 Model: {model_name}")
print(f"📋 Version: {latest_version.version}")
print(f"🆔 Version ID: {latest_version.version}")
print(f"📅 Created: {latest_version.creation_timestamp}")
print(f"🔗 Source: {latest_version.source}")
print("\n" + "="*50)

# Get model metadata
model_version = client.get_model_version(model_name, latest_version.version)

print("\n📊 MODEL METADATA:")
print("-" * 30)

# Check if metadata exists
if hasattr(model_version, 'tags') and model_version.tags:
    print("📋 Tags:")
    for key, value in model_version.tags.items():
        print(f"  • {key}: {value}")

print("\n📁 MODEL ARTIFACTS AND METADATA:")
print("-" * 40)

# Load the model to get its metadata
model_uri = f"models:/{model_name}/{latest_version.version}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Get model info
model_info = client.get_model_version(model_name, latest_version.version)

print(f"🎯 Model URI: {model_uri}")
print(f"🔄 Current Stage: {model_info.current_stage}")
print(f"📝 Description: {model_info.description}")

# Try to get run info to see run metadata
run_id = model_info.run_id
if run_id:
    run = client.get_run(run_id)
    print(f"\n🏃 RUN INFORMATION:")
    print(f"  • Run ID: {run_id}")
    print(f"  • Experiment ID: {run.info.experiment_id}")
    
    print(f"\n📊 RUN METRICS:")
    for key, value in run.data.metrics.items():
        print(f"  • {key}: {value:.4f}")
    
    print(f"\n⚙️  RUN PARAMETERS:")
    for key, value in run.data.params.items():
        print(f"  • {key}: {value}")
    
    print(f"\n🏷️  RUN TAGS:")
    for key, value in run.data.tags.items():
        if not key.startswith('mlflow.'):  # Skip internal MLflow tags
            print(f"  • {key}: {value}")

# Try to access model metadata directly from the model artifacts
try:
    import os
    import yaml
    
    # Download model artifacts to check metadata
    artifact_path = client.download_artifacts(run_id, "iris_model")
    
    # Check if MLmodel file exists and read it
    mlmodel_path = os.path.join(artifact_path, "MLmodel")
    if os.path.exists(mlmodel_path):
        print(f"\n📄 MODEL FILE METADATA:")
        with open(mlmodel_path, 'r') as f:
            mlmodel_content = yaml.safe_load(f)
            
        if 'metadata' in mlmodel_content:
            print("  Model Metadata found in MLmodel file:")
            metadata = mlmodel_content['metadata']
            for key, value in metadata.items():
                if isinstance(value, list):
                    print(f"  • {key}: {', '.join(map(str, value))}")
                else:
                    print(f"  • {key}: {value}")
        else:
            print("  No custom metadata found in MLmodel file")
            
        # Show model signature
        if 'signature' in mlmodel_content:
            print(f"\n📋 MODEL SIGNATURE:")
            signature = mlmodel_content['signature']
            if 'inputs' in signature:
                print("  Inputs:")
                for input_spec in signature['inputs']:
                    print(f"    • {input_spec.get('name', 'unnamed')}: {input_spec.get('type', 'unknown')}")
            if 'outputs' in signature:
                print("  Outputs:")
                for output_spec in signature['outputs']:
                    print(f"    • {output_spec.get('name', 'unnamed')}: {output_spec.get('type', 'unknown')}")
                    
except Exception as e:
    print(f"\n⚠️  Could not access model artifacts: {e}")

print("\n" + "="*50)
print("✅ Metadata check complete!") 