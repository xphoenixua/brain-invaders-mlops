import os
import argparse
import datetime
import io
import joblib
import numpy as np
import pandas as pd
import tempfile

import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix


# constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio-server:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_PROCESSED_BUCKET = os.getenv("MINIO_PROCESSED_BUCKET")
MINIO_MODELS_BUCKET = os.getenv("MINIO_MODELS_BUCKET")

MODEL_CONFIG = {
    "type": "LDA", # can be "LDA" or "Ridge"
    "lda_solver": "lsqr",
    "lda_shrinkage": "auto",
    "ridge_alphas": np.logspace(-3, 3, 7)
}

TARGET_CLASS_LABEL_CODE = 2
MLFLOW_TRACKING_URI_FALLBACK = "http://mlflow-server:5000"
MLFLOW_EXPERIMENT_NAME = "P300_BCI_Training"
MLFLOW_REGISTERED_MODEL_NAME = "P300-Classifier"

# this is crucial for mlflow.log_artifacts to work with minio
# for some reason, mlflow doesn't pick up the environment variables set in the minio-init container
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", MINIO_ENDPOINT)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", MINIO_ACCESS_KEY)
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", MINIO_SECRET_KEY)

# s3 client initialization
def create_s3_client(endpoint, access_key, secret_key):
    """creates a boto3 s3 client with specified configurations"""
    return boto3.client(
        's3', endpoint_url=endpoint, aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version='s3v4', region_name='us-east-1'),
        use_ssl=False, verify=False
    )

s3_client = create_s3_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)


def get_available_processed_subject_ids(s3_client, bucket_name) -> list[int]:
    """scans the processed bucket and returns sorted list of available subject ids"""
    subject_ids = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Delimiter='/'):
            for prefix_obj in page.get('CommonPrefixes', []):
                folder_name = prefix_obj['Prefix'].strip('/')
                if folder_name.startswith("subject_"):
                    try: subject_ids.add(int(folder_name.split('_')[1]))
                    except (ValueError, IndexError): continue # ignore malformed folder names
        return sorted(list(subject_ids))
    except Exception as e:
        print(f"error listing processed subjects in {bucket_name}: {e}")
        return []


def download_subject_processed_data(s3_client, bucket_name, subject_id):
    """downloads processed features and labels for a single subject from minio"""
    print(f"  downloading processed data for subject {subject_id:02d} from {bucket_name}")
    features_key = f"subject_{subject_id:02d}/features.parquet"
    labels_key = f"subject_{subject_id:02d}/labels.parquet"

    try:
        features_obj = s3_client.get_object(Bucket=bucket_name, Key=features_key)
        x_df = pd.read_parquet(io.BytesIO(features_obj['Body'].read()))

        labels_obj = s3_client.get_object(Bucket=bucket_name, Key=labels_key)
        y_df = pd.read_parquet(io.BytesIO(labels_obj['Body'].read()))
        return x_df.values, y_df['label'].values
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"    warning: data not found for subject {subject_id:02d}")
        else:
            print(f"    error downloading data for subject {subject_id:02d}: {e}")
        return None, None
    except Exception as e:
        print(f"    unexpected error downloading for subject {subject_id:02d}: {e}")
        return None, None


def train_model_sklearn(x_train, y_train, model_config, scaler_instance=None):
    """trains an sklearn model with optional scaling"""
    x_train_scaled = scaler_instance.fit_transform(x_train) if scaler_instance else x_train
    print(f"    training {model_config['type']} model")

    if model_config['type'] == "LDA":
        model = LinearDiscriminantAnalysis(solver=model_config['lda_solver'], shrinkage=model_config['lda_shrinkage'])
    elif model_config['type'] == "Ridge":
        model = RidgeClassifierCV(alphas=model_config['ridge_alphas'], store_cv_values=True)
    else:
        raise ValueError(f"unsupported model type: {model_config['type']}")

    model.fit(x_train_scaled, y_train)
    if model_config['type'] == "Ridge": print(f"    ridge best alpha: {model.alpha_}")
    return model, x_train_scaled


def evaluate_model_sklearn(model, x_test, y_test, scaler_instance=None):
    """evaluates an sklearn model and returns performance metrics"""
    x_test_scaled = scaler_instance.transform(x_test) if scaler_instance else x_test

    y_pred = model.predict(x_test_scaled)
    y_test_binary = np.where(y_test == TARGET_CLASS_LABEL_CODE, 1, 0)
    y_pred_binary = np.where(y_pred == TARGET_CLASS_LABEL_CODE, 1, 0)

    auc, accuracy, kappa = None, 0, 0
    conf_matrix = np.zeros((2, 2), dtype=int)

    try:
        if hasattr(model, "predict_proba"):
            target_class_idx = list(model.classes_).index(TARGET_CLASS_LABEL_CODE)
            y_pred_proba_target = model.predict_proba(x_test_scaled)[:, target_class_idx]
            if len(np.unique(y_test_binary)) > 1: auc = roc_auc_score(y_test_binary, y_pred_proba_target)
        elif hasattr(model, "decision_function"):
            decision_scores = model.decision_function(x_test_scaled)
            if len(np.unique(y_test_binary)) > 1: auc = roc_auc_score(y_test_binary, decision_scores)

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        kappa = cohen_kappa_score(y_test_binary, y_pred_binary)
        conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
    except Exception as e:
        print(f"warning: metric calculation issue: {e}")

    print(f"    auc: {auc if auc is not None else 'n/a'}")
    print(f"    accuracy: {accuracy:.4f}")
    print(f"    kappa: {kappa:.4f}")
    print(f"    confusion matrix (target=1, nontarget=0):\n{conf_matrix}")

    return {
        "auc": auc,
        "accuracy": accuracy,
        "kappa": kappa,
        "conf_matrix_tn": conf_matrix[0, 0],
        "conf_matrix_fp": conf_matrix[0, 1],
        "conf_matrix_fn": conf_matrix[1, 0],
        "conf_matrix_tp": conf_matrix[1, 1]
    }


def main_training_pipeline(training_subjects_percentage: float, internal_val_split_ratio=0.8, 
                           cli_model_version=None, auto_set_alias=True):
    """
    runs the full training pipeline, including data download, model training,
    evaluation, and logging to W&B and MinIO
    """
    available_subject_ids = get_available_processed_subject_ids(s3_client, MINIO_PROCESSED_BUCKET)
    if not available_subject_ids:
        print("no processed subjects found in minio. exiting training")
        return

    num_subjects_to_use = max(1, int(len(available_subject_ids) * training_subjects_percentage)) if available_subject_ids else 0
    if num_subjects_to_use == 0:
        print("no subjects to use for training. exiting")
        return

    subject_ids = available_subject_ids[:num_subjects_to_use]

    model_version_tag = cli_model_version if cli_model_version else f"v{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # mlflow setup
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_FALLBACK)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    print(f"mlflow tracking uri confirmed by client: {mlflow.get_tracking_uri()}")
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        try:
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME) # uses default artifact root from server
            print(f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' created with id: {experiment_id}")
            mlflow.set_experiment(experiment_id=experiment_id)
        except mlflow.exceptions.MlflowException as e_create:
            print(f"MLflow: error creating experiment '{MLFLOW_EXPERIMENT_NAME}': {e_create}.")
    else:
        print(f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' already exists with id: {experiment.experiment_id}")
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)

    print(f"starting training pipeline for {len(subject_ids)} subjects: {subject_ids}")
    all_x, all_y = [], []
    for sub_id in subject_ids:
        x_sub, y_sub = download_subject_processed_data(s3_client, MINIO_PROCESSED_BUCKET, sub_id)
        if x_sub is not None and y_sub is not None and x_sub.shape[0] > 0:
            all_x.append(x_sub)
            all_y.append(y_sub)
        else:
            print(f"    skipping subject {sub_id:02d} due to missing or empty data")
    if not all_x:
        print("no data loaded for selected subjects. exiting")
        return

    x_combined = np.vstack(all_x)
    y_combined = np.concatenate(all_y)
    print(f"combined data: x={x_combined.shape}, labels: {np.unique(y_combined)}")
    if len(np.unique(y_combined)) < 2:
        print("need 2 or more classes for training. exiting")
        return

    x_train, x_val, y_train, y_val = train_test_split(
        x_combined, y_combined, test_size=(1.0 - internal_val_split_ratio),
        stratify=y_combined, random_state=42
    )
    print(f"internal train set: {x_train.shape}, internal validation set: {x_val.shape}")

    # start MLflow run
    with mlflow.start_run(run_name=f"train_run_{model_version_tag}") as run:
        run_id = run.info.run_id
        print(f"mlflow run started. run id: {run_id}")

        mlflow_params = {
            "training_subjects_percentage": training_subjects_percentage, "internal_val_split_ratio": internal_val_split_ratio,
            "model_type": MODEL_CONFIG['type'], "num_subjects_trained_on": len(subject_ids),
            "subject_ids_trained_on_list_str": str(subject_ids),
            "total_processed_subjects_available": len(available_subject_ids),
            "generated_model_version_tag": model_version_tag,
            "internal_training_set_size": x_train.shape[0], "internal_validation_set_size": x_val.shape[0]
        }

        if MODEL_CONFIG['type'] == "LDA": mlflow_params.update({"lda_solver": MODEL_CONFIG.get("lda_solver"), 
                                                                "lda_shrinkage": MODEL_CONFIG.get("lda_shrinkage")})
        elif MODEL_CONFIG['type'] == "Ridge": mlflow_params["ridge_alphas"] = str(MODEL_CONFIG.get("ridge_alphas"))
        
        mlflow.log_params(mlflow_params)


        scaler = StandardScaler()
        trained_model, x_train_scaled = train_model_sklearn(x_train, y_train, MODEL_CONFIG, scaler_instance=scaler)
        print(f"model classes after fitting: {trained_model.classes_}")

        print("\ninternal validation set evaluation")
        val_metrics_dict = evaluate_model_sklearn(trained_model, x_val, y_val, scaler_instance=scaler)

        mlflow.log_metrics(val_metrics_dict)

        # infer signature for MLflow model
        model_signature = None
        input_example_data = None

        if x_train_scaled.shape[0] > 0:
            num_features = x_train_scaled.shape[1]
            input_schema = Schema([ColSpec("double", f"feature_{i}") for i in range(num_features)])
            if hasattr(trained_model, "predict_proba"):
                output_schema = Schema([ColSpec("double", "probability_class_0"), 
                                        ColSpec("double", "probability_class_1")])
            else:
                output_schema = Schema([ColSpec("integer", "prediction")])
            model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            input_example_data = pd.DataFrame(x_train_scaled[:min(5, x_train_scaled.shape[0]), :], 
                                              columns=[f"feature_{i}" for i in range(num_features)])
            print("mlflow: model signature defined explicitly.")
        else: 
            print("mlflow: skipping model signature, no training data.")

        registered_model_info = None
        with tempfile.TemporaryDirectory() as tmp_path:
            model_saved_path = os.path.join(tmp_path, "model_payload")
            scaler_saved_path = os.path.join(tmp_path, "scaler_payload")

            mlflow.sklearn.save_model(trained_model, path=model_saved_path, signature=model_signature, input_example=input_example_data)
            # save scaler without signature/input_example as it's a preprocessor
            joblib.dump(scaler, scaler_saved_path)

            registered_model_info = mlflow.pyfunc.log_model(
                python_model=mlflow.pyfunc.PythonModel(),
                artifact_path="p300-model-with-scaler",
                artifacts={ "model_dir": model_saved_path, "scaler_file": scaler_saved_path },
                registered_model_name=MLFLOW_REGISTERED_MODEL_NAME
            )
        new_model_version = registered_model_info.registered_model_version
        print(f"mlflow: model and scaler logged. registered '{MLFLOW_REGISTERED_MODEL_NAME}' version '{new_model_version}'.")

        if auto_set_alias:
            try:
                print(f"setting alias 'challenger' for model '{MLFLOW_REGISTERED_MODEL_NAME}' version '{new_model_version}'.")
                client.set_registered_model_alias(
                    name=MLFLOW_REGISTERED_MODEL_NAME,
                    alias='challenger',
                    version=new_model_version
                )
                print(f"alias 'challenger' set successfully.")
            except Exception as e_alias:
                print(f"error setting model alias: {e_alias}")

    print(f"finished training pipeline. run id: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train p300 model from minio, log to w&b.")
    parser.add_argument("--training_subjects_percentage", type=float, default=1.0,
                        help="percentage of available processed subjects to use for training (0.0 to 1.0). defaulted to 1.0 (all available)")
    parser.add_argument("--internal_val_split_ratio", type=float, default=0.8,
                        help="ratio for splitting the selected subjects' data into internal train/validation. defaulted to 0.8")
    parser.add_argument("--model_version_tag", type=str, default=None, 
                        help="optional tag for run name (not MLflow model version)")
    parser.add_argument("--auto_set_alias", default=True, 
                        help="automatically set the 'challenger' alias for the new model version")
    args = parser.parse_args()

    main_training_pipeline(
        training_subjects_percentage=args.training_subjects_percentage,
        internal_val_split_ratio=args.internal_val_split_ratio,
        cli_model_version=args.model_version_tag,
        auto_set_alias=args.auto_set_alias
    )
    