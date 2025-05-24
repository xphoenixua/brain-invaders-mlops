import os
import joblib
import numpy as np
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_MODELS_BUCKET = os.getenv("MINIO_MODELS_BUCKET")
DEFAULT_ACTIVE_MODEL_VERSION = os.getenv("DEFAULT_ACTIVE_MODEL_VERSION", "v_initial_placeholder")
ACTIVE_POINTER_FILENAME = "active_model_pointer.txt"

MODEL_OUTPUT_MAPPING = {
    1: {"label_code": 1, "class_name": "NonTarget"}, # model predicts 1 (original NonTarget)
    2: {"label_code": 2, "class_name": "Target"}    # model predicts 2 (original Target)
}
TARGET_CLASS_ORIGINAL_CODE = 2 # original label code for target


# global variables
app = FastAPI(title="P300 BCI prediction service")
model = None
scaler = None
current_model_version = None
s3_client = None


class EpochFeatures(BaseModel):
    features: list[float]


def create_s3_client(endpoint, access_key, secret_key):
    """creates a boto3 s3 client with specified configurations"""
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version='s3v4', region_name='us-east-1'),
        use_ssl=False,
        verify=False
    )


def download_and_load_joblib_file(bucket_name, s3_key):
    """downloads a joblib file from s3 and loads it"""
    local_path = f"/tmp/{os.path.basename(s3_key)}"
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        obj = joblib.load(local_path)
        print(f"downloaded and loaded {s3_key.split('/')[-1]} successfully")
        return obj
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"error: file not found in minio: '{s3_key}'")
        else:
            print(f"client error downloading '{s3_key}': {e}")
    except Exception as e:
        print(f"unexpected error loading '{s3_key}': {e}")
    finally:
        if os.path.exists(local_path): os.remove(local_path)
    return None


async def load_active_model_from_minio():
    """loads the active model and scaler from minio based on the pointer file"""
    global model, scaler, current_model_version

    print("attempting to load active model from minio")
    active_version_str = DEFAULT_ACTIVE_MODEL_VERSION

    # try to get the active model version from the pointer file
    try:
        response = s3_client.get_object(Bucket=MINIO_MODELS_BUCKET, Key=ACTIVE_POINTER_FILENAME)
        active_version_str = response['Body'].read().decode('utf-8').strip()
        print(f"found active model version pointer: '{active_version_str}'")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"warning: active model pointer '{ACTIVE_POINTER_FILENAME}' not found. using default: '{DEFAULT_ACTIVE_MODEL_VERSION}'")
        else:
            print(f"error reading active model pointer: {e}")
    except Exception as e:
        print(f"unexpected error reading active model pointer: {e}")

    if active_version_str == current_model_version and model is not None:
        print(f"model version '{active_version_str}' is already loaded. skipping reload")
        return

    if active_version_str == DEFAULT_ACTIVE_MODEL_VERSION and DEFAULT_ACTIVE_MODEL_VERSION == "v_initial_placeholder":
        print("no valid active model specified, pointer points to placeholder or is default. model not loaded")
        return

    # paths for model and scaler within the versioned folder in s3
    s3_model_key = f"{active_version_str}/model.joblib"
    s3_scaler_key = f"{active_version_str}/scaler.joblib"

    model_new = download_and_load_joblib_file(MINIO_MODELS_BUCKET, s3_model_key)
    scaler_new = download_and_load_joblib_file(MINIO_MODELS_BUCKET, s3_scaler_key)

    if model_new is None or scaler_new is None:
        print(f"failed to load model or scaler for version '{active_version_str}'. model loading failed")
        model = None
        scaler = None
        current_model_version = f"load_failed_{active_version_str}"
        return

    model = model_new
    scaler = scaler_new
    current_model_version = active_version_str
    print(f"successfully loaded model version: {current_model_version}")
    if hasattr(model, 'classes_'): print(f"  loaded model classes_: {model.classes_}")


# startup event: load model and scaler from minio
@app.on_event("startup")
async def startup_event():
    global s3_client
    print("fastapi application startup")

    s3_client = create_s3_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
    await load_active_model_from_minio()


# prediction endpoint
@app.post("/predict/")
async def predict_p300(epoch_data: EpochFeatures):
    """predicts p300 event for given eeg features"""
    if model is None:
        raise HTTPException(status_code=503, detail=f"model (version: {current_model_version}) not loaded or failed to load. serving unavailable")

    try:
        feature_vector = np.array(epoch_data.features).astype(np.float64).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector) if scaler is not None else feature_vector

        prediction_internal = model.predict(feature_vector_scaled)[0]
        predicted_info = MODEL_OUTPUT_MAPPING.get(prediction_internal)

        if predicted_info is None:
            print(f"warning: unknown internal prediction value: {int(prediction_internal)}")
            raise HTTPException(status_code=500, detail=f"unknown internal prediction value: {int(prediction_internal)}")

        predicted_label_code = predicted_info["label_code"]
        predicted_class_name = predicted_info["class_name"]
        target_probability = None

        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(feature_vector_scaled)[0]
            try:
                target_class_idx_in_model_classes = list(model.classes_).index(TARGET_CLASS_ORIGINAL_CODE)
                target_probability = float(prediction_proba[target_class_idx_in_model_classes])
            except (ValueError, IndexError) as e:
                print(f"warning: target class code {TARGET_CLASS_ORIGINAL_CODE} not in model classes {model.classes_} or index issue: {e}")
        elif hasattr(model, "decision_function"):
            target_probability = float(model.decision_function(feature_vector_scaled)[0])

        return {
            "model_version": current_model_version,
            "predicted_label_code": predicted_label_code,
            "predicted_class_name": predicted_class_name,
            "probability_target": target_probability
        }
    except Exception as e:
        print(f"error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"internal server error during prediction: {e}")


# health check endpoint
@app.get("/health")
async def health_check():
    """provides health status of the service"""
    if model is not None:
        return {"status": "ok", "model_version": current_model_version, "detail": "model loaded"}
    else:
        # service is up, but model is not ready, then still return 200 for basic health
        return {"status": "degraded", "model_version": current_model_version, "detail": "service running, model not loaded"}    