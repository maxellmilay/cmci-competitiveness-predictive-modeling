import mlflow
from mlflow.models import infer_signature
import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    try:
        # Load the Iris dataset
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        
        # Convert to DataFrame for better schema inference
        feature_names = iris.feature_names
        target_names = iris.target_names
        X_df = pd.DataFrame(X, columns=feature_names)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.1, random_state=42
        )

        # Define the model hyperparameters
        params = {
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 8888,
        }

        # Train the model
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        # Predict on the test set
        y_pred = lr.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Set our tracking server uri for logging (use env var or default)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
        mlflow.set_tracking_uri(uri=tracking_uri)

        # Create a new MLflow Experiment
        experiment_name = "MLflow Quickstart"
        mlflow.set_experiment(experiment_name)

        # Start an MLflow run with a descriptive name
        with mlflow.start_run(run_name="iris_logistic_regression") as run:
            # Log the hyperparameters
            mlflow.log_params(params)

            # Log all metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log additional metadata and metrics as tags for the model
            mlflow.set_tag("Training Info", "Basic LR model for iris data")
            mlflow.set_tag("model_type", "LogisticRegression")
            mlflow.set_tag("dataset", "iris")
            mlflow.set_tag("features", ", ".join(feature_names))
            mlflow.set_tag("target_classes", ", ".join(target_names))
            mlflow.set_tag("accuracy", f"{accuracy:.4f}")
            mlflow.set_tag("f1_score", f"{f1:.4f}")

            # Create proper signature with DataFrame for clear schema
            # Get predictions and create a proper output schema with class names
            predictions = lr.predict(X_train)
            # Map predictions to class names for better schema
            prediction_classes = [target_names[pred] for pred in predictions]
            predictions_df = pd.DataFrame({
                "predicted_class_index": predictions,
                "predicted_class_name": prediction_classes
            })
            signature = infer_signature(X_train, predictions_df)

            # Log the model with clear metadata
            model_info = mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="iris_model",
                signature=signature,
                input_example=X_train.head(3),  # Use DataFrame head for clear example
                registered_model_name="iris-classification-model",
                metadata={
                    "features": [str(name) for name in feature_names],
                    "target_classes": [str(name) for name in target_names],
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }
            )

            # Add metadata as model version tags (visible in UI)
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Get the model version that was just created
            model_name = "iris-classification-model"
            latest_versions = client.search_model_versions(f"name='{model_name}'")
            latest_version = max(latest_versions, key=lambda x: int(x.version))
            
            # Set model version tags
            client.set_model_version_tag(model_name, latest_version.version, "accuracy", f"{accuracy:.4f}")
            client.set_model_version_tag(model_name, latest_version.version, "precision", f"{precision:.4f}")
            client.set_model_version_tag(model_name, latest_version.version, "recall", f"{recall:.4f}")
            client.set_model_version_tag(model_name, latest_version.version, "f1_score", f"{f1:.4f}")
            client.set_model_version_tag(model_name, latest_version.version, "features", ", ".join(feature_names))
            client.set_model_version_tag(model_name, latest_version.version, "target_classes", ", ".join(target_names))
            client.set_model_version_tag(model_name, latest_version.version, "model_type", "LogisticRegression")
            client.set_model_version_tag(model_name, latest_version.version, "dataset", "iris")

            # Print model info for verification
            print(f"Model saved in run {run.info.run_id}")
            print(f"Experiment: {experiment_name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Model URI: {model_info.model_uri}")
            print(f"Features: {', '.join(feature_names)}")
            print(f"Target Classes: {', '.join(target_names)}")
            
    except Exception as e:
        print(f"Error during MLflow logging: {str(e)}")
        raise


if __name__ == "__main__":
    main()
