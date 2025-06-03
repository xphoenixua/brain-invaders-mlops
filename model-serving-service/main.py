import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import time
import pandas as pd
import joblib

# same thing as in training service
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", os.getenv("MINIO_ENDPOINT"))
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ACCESS_KEY"))
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_SECRET_KEY"))

# model serving configuration
SERVING_MODEL_ALIAS = os.getenv("SERVING_MODEL_ALIAS", "champion") # e.g., 'champion' or 'challenger'
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "P300-Classifier")
SERVING_MODEL_URI = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{SERVING_MODEL_ALIAS}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

MODEL_OUTPUT_MAPPING = {
    1: {"label_code": 1, "class_name": "NonTarget"}, # model predicts 1 (original NonTarget)
    2: {"label_code": 2, "class_name": "Target"}    # model predicts 2 (original Target)
}
TARGET_CLASS_ORIGINAL_CODE = 2 # original label code for target


# global variables
app = FastAPI(title="P300 BCI prediction service")
model_pipeline_components = None
current_model_uri_loaded = None
current_model_version_loaded = None


class PythonModelWrapper(mlflow.pyfunc.PythonModel):
    """
    A custom wrapper to ensure the scaler is applied before prediction.
    """
    def __init__(self, model, scaler):
        self._model = model
        self._scaler = scaler

    def predict(self, context, model_input):
        scaled_features = self._scaler.transform(model_input.values)
        predictions = self._model.predict(scaled_features)
        probabilities = self._model.predict_proba(scaled_features)
        return {"predictions": predictions, "probabilities": probabilities}

class EpochFeatures(BaseModel):
    features: list[float]


async def load_model_mlflow(model_uri, max_retries=5, delay_seconds=10):
    global model_pipeline_components, current_model_uri_loaded, current_model_version_loaded
    print(f"attempting to load model from MLflow registry using URI: {model_uri}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    for attempt in range(max_retries):
        try:
            # first, resolve the alias to a specific model version to get its run_id or source.
            try:
                model_name_from_uri, alias_from_uri = model_uri.split("@")[0].split("/")[-1], model_uri.split("@")[-1]
                version_info = client.get_model_version_by_alias(model_name_from_uri, alias_from_uri)
                pyfunc_source_uri = version_info.source # this is the MinIO URI of the pyfunc model artifact
                current_model_version_loaded = version_info.version
                print(f"resolved alias '{alias_from_uri}' to version '{current_model_version_loaded}' of model '{model_name_from_uri}'.")
            except Exception as e_resolve:
                print(f"could not resolve model URI '{model_uri}' via alias: {e_resolve}")
                raise e_resolve
            

            print(f"downloading entire pyfunc model (scaler + classifier) artifact from: {pyfunc_source_uri}")
            # points to the local pyfunc artifact downloaded from MinIO to a local path
            local_pyfunc_root = mlflow.artifacts.download_artifacts(
                artifact_uri=pyfunc_source_uri
            )

            artifacts_subdir_path = os.path.join(local_pyfunc_root, "artifacts")
            local_model_path = os.path.join(artifacts_subdir_path, "model_payload")
            local_scaler_path = os.path.join(artifacts_subdir_path, "scaler_payload")
            print(f"contents of 'artifacts' subdir ({artifacts_subdir_path}): {os.listdir(artifacts_subdir_path)}")

            if not os.path.isdir(local_model_path):
                raise FileNotFoundError(f"expected model directory not found after download at {local_model_path}")
            if not os.path.isfile(local_scaler_path):
                raise FileNotFoundError(f"expected scaler file not found after download at {local_scaler_path}")

            # load the model and scaler
            sklearn_model = mlflow.sklearn.load_model(model_uri=local_model_path)
            print(f"scikit-learn model loaded successfully from {local_model_path}")
            scaler = joblib.load(local_scaler_path)
            print(f"scaler loaded successfully from {local_scaler_path}")
            
            model_pipeline_components = {"model": sklearn_model, "scaler": scaler}
            current_model_uri_loaded = model_uri
            
            print(f"successfully loaded model version {current_model_version_loaded} (from {model_uri}) and scaler.")
            app.extra["model_version_details"] = f"v{current_model_version_loaded} (source: {pyfunc_source_uri})"
            return 
        
            
        except Exception as e:
            print(f"error loading model (attempt {attempt + 1}/{max_retries}): {e}")
            import traceback
            traceback.print_exc()
            if attempt < max_retries - 1:
                print(f"retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("failed to load model after multiple retries.")
                model_pipeline_components = None
                current_model_uri_loaded = f"load_failed_{model_uri}"
                current_model_version_loaded = "load_failed"
                app.extra["model_version_details"] = "load_failed"


@app.on_event("startup")
async def startup_event():
    print(f"fastapi application startup. loading model: {SERVING_MODEL_URI}")
    app.extra = {} # to store model version details
    await load_model_mlflow(SERVING_MODEL_URI)


# prediction endpoint
@app.post("/predict/")
async def predict_p300(epoch_data: EpochFeatures):
    """predicts p300 event for given eeg features"""
    if model_pipeline_components is None or model_pipeline_components.get("model") is None:
        raise HTTPException(status_code=503, detail=f"model (uri: {current_model_uri_loaded}) not loaded. serving unavailable.")

    try:
        model = model_pipeline_components["model"]
        scaler = model_pipeline_components["scaler"]
        
        feature_vector_np = np.array(epoch_data.features).reshape(1, -1)
        scaled_features = scaler.transform(feature_vector_np)
        
        prediction_internal_label = model.predict(scaled_features)[0]

        model_output_mapping_local = {
            cls_label: {"label_code": int(cls_label), "class_name": "Target" if int(cls_label) == TARGET_CLASS_ORIGINAL_CODE else "NonTarget"}
            for cls_label in model.classes_
        }
        
        predicted_info = model_output_mapping_local.get(int(prediction_internal_label))

        if predicted_info is None:
            raise HTTPException(status_code=500, detail=f"unknown internal prediction value: {prediction_internal_label}")

        predicted_label_code = predicted_info["label_code"]
        predicted_class_name = predicted_info["class_name"]
        target_probability = None

        try:
            proba_all_classes = model.predict_proba(scaled_features)[0]
            target_class_idx_in_model = list(model.classes_).index(TARGET_CLASS_ORIGINAL_CODE)
            target_probability = float(proba_all_classes[target_class_idx_in_model])
        except Exception as e_proba:
            print(f"warning: could not get target probability: {e_proba}")
        
        return {
            "model_version_loaded": app.extra.get("model_version_details", current_model_uri_loaded),
            "predicted_label_code": predicted_label_code,
            "predicted_class_name": predicted_class_name,
            "probability_target": target_probability
        }
    
    except Exception as e:
        print(f"error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"internal server error: {e}")


@app.get("/health")
async def health_check():
    model_status = "loaded" if model_pipeline_components is not None else "not_loaded"
    return {
        "status": "ok" if model_status == "loaded" else "degraded",
        "model_uri_configured": SERVING_MODEL_URI,
        "model_uri_actually_loaded_from_run": current_model_uri_loaded if model_status == "loaded" else "N/A", # this will be the alias URI
        "model_version_loaded": current_model_version_loaded,
        "model_details_from_registry": app.extra.get("model_version_details", "N/A")
    }
