import mlflow
from mlflow.tracking import MlflowClient
import tempfile
import os
import yaml

mlflow.set_tracking_uri("http://127.0.0.1:8080")
client = MlflowClient()

model_name = "iris-classification-model"
latest_version = client.search_model_versions(f"name='{model_name}'")[0]

print("=" * 60)
print("🔍 DIFFERENT WAYS TO STORE/ACCESS MODEL METADATA")
print("=" * 60)

print("\n1️⃣ MODEL VERSION TAGS (visible in Model Registry UI):")
print("-" * 50)
try:
    model_version = client.get_model_version(model_name, latest_version.version)
    if hasattr(model_version, 'tags') and model_version.tags:
        for key, value in model_version.tags.items():
            print(f"   📋 {key}: {value}")
    else:
        print("   ❌ No model version tags found")
except Exception as e:
    print(f"   ❌ Error accessing tags: {e}")

print("\n2️⃣ MODEL ARTIFACTS METADATA (stored in model files):")
print("-" * 50)
try:
    # Load model and access metadata
    model_uri = f"models:/{model_name}/{latest_version.version}"
    
    # Download model artifacts to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get the run ID and download artifacts
        run_id = latest_version.run_id
        artifact_path = client.download_artifacts(run_id, "iris_model", temp_dir)
        
        # Read MLmodel file
        mlmodel_path = os.path.join(artifact_path, "MLmodel")
        if os.path.exists(mlmodel_path):
            with open(mlmodel_path, 'r') as f:
                mlmodel_content = yaml.safe_load(f)
            
            if 'metadata' in mlmodel_content:
                print("   📁 Found metadata in model artifacts:")
                for key, value in mlmodel_content['metadata'].items():
                    if isinstance(value, list):
                        print(f"      • {key}: {', '.join(map(str, value))}")
                    else:
                        print(f"      • {key}: {value}")
            else:
                print("   ❌ No metadata found in MLmodel file")
        else:
            print("   ❌ MLmodel file not found")
            
except Exception as e:
    print(f"   ❌ Error accessing model artifacts: {e}")

print("\n3️⃣ SOURCE RUN METADATA (from the training run):")
print("-" * 50)
try:
    run_id = latest_version.run_id
    run = client.get_run(run_id)
    
    print("   📊 Metrics:")
    for key, value in run.data.metrics.items():
        print(f"      • {key}: {value:.4f}")
    
    print("   ⚙️  Parameters:")
    for key, value in run.data.params.items():
        print(f"      • {key}: {value}")
    
    print("   🏷️  Tags:")
    for key, value in run.data.tags.items():
        if not key.startswith('mlflow.'):
            print(f"      • {key}: {value}")
            
except Exception as e:
    print(f"   ❌ Error accessing run metadata: {e}")

print("\n4️⃣ PROGRAMMATIC ACCESS TO MODEL METADATA:")
print("-" * 50)
try:
    # This is how you'd access metadata in production code
    import mlflow.sklearn
    
    model_uri = f"models:/{model_name}/{latest_version.version}"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    # Check if the model has metadata attribute
    if hasattr(loaded_model, 'metadata'):
        print("   ✅ Model has metadata attribute")
        print(f"   📋 Metadata: {loaded_model.metadata}")
    else:
        print("   ❌ No metadata attribute on loaded model")
        
    # Access via pyfunc for metadata
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    if hasattr(pyfunc_model, 'metadata'):
        print(f"   📋 PyFunc Metadata: {pyfunc_model.metadata}")
    
except Exception as e:
    print(f"   ❌ Error loading model: {e}")

print("\n" + "=" * 60)
print("💡 RECOMMENDATIONS:")
print("=" * 60)
print("""
✅ BEST PRACTICES:

1. 🎯 For PERFORMANCE METRICS → Use Run Metrics (always visible in experiments)
2. 📋 For KEY MODEL INFO → Use Model Version Tags (visible in model registry)  
3. 📁 For DETAILED METADATA → Store in model artifacts metadata
4. 📝 For HUMAN DESCRIPTION → Use model version description
5. 🔗 For TRACEABILITY → Link via Source Run (automatically done)

🎯 The HYBRID APPROACH:
- Store detailed metadata in model artifacts (for programmatic access)
- Store key metrics as model version tags (for UI visibility)
- Keep full training context in the source run
""") 