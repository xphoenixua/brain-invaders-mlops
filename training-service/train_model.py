import os
import argparse
import datetime
import io
import joblib
import numpy as np
import pandas as pd
import wandb

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
TARGET_CLASS_LABEL_CODE = 2 # original label code for 'target'


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
    return model


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


def main_training_pipeline(training_subjects_percentage: float, internal_val_split_ratio=0.8, cli_model_version=None):
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

    subject_ids_for_this_run = available_subject_ids[:num_subjects_to_use]

    model_version_to_use = cli_model_version if cli_model_version else f"v{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    wandb.init(
        project="p300-bci-mlops",
        name=f"train_{model_version_to_use}_on_{num_subjects_to_use}subjects",
        job_type="training"
    )
    wandb.config.training_subjects_percentage = training_subjects_percentage
    wandb.config.internal_val_split_ratio = internal_val_split_ratio
    wandb.config.model_type = MODEL_CONFIG['type']
    wandb.config.subject_ids_trained_on = subject_ids_for_this_run
    wandb.config.num_subjects_trained_on = len(subject_ids_for_this_run)
    wandb.config.total_processed_subjects_available = len(available_subject_ids)
    wandb.config.model_version_saved = model_version_to_use

    print(f"starting training pipeline for {len(subject_ids_for_this_run)} subjects: {subject_ids_for_this_run}")
    all_x, all_y = [], []
    for sub_id in subject_ids_for_this_run:
        x_sub, y_sub = download_subject_processed_data(s3_client, MINIO_PROCESSED_BUCKET, sub_id)
        if x_sub is not None and y_sub is not None and x_sub.shape[0] > 0:
            all_x.append(x_sub)
            all_y.append(y_sub)
        else:
            print(f"    skipping subject {sub_id:02d} due to missing or empty data")
    if not all_x:
        print("no data loaded for selected subjects. exiting")
        wandb.finish()
        return

    x_combined = np.vstack(all_x)
    y_combined = np.concatenate(all_y)
    print(f"combined data: x={x_combined.shape}, labels: {np.unique(y_combined)}")
    if len(np.unique(y_combined)) < 2:
        print("need 2 or more classes for training. exiting")
        wandb.finish()
        return

    x_train, x_val, y_train, y_val = train_test_split(
        x_combined, y_combined, test_size=(1.0 - internal_val_split_ratio),
        stratify=y_combined, random_state=42
    )
    print(f"internal train set: {x_train.shape}, internal validation set: {x_val.shape}")
    wandb.config.internal_training_set_size = x_train.shape[0]
    wandb.config.internal_validation_set_size = x_val.shape[0]

    scaler = StandardScaler()
    trained_model = train_model_sklearn(x_train, y_train, MODEL_CONFIG, scaler_instance=scaler)
    print(f"model classes after fitting: {trained_model.classes_}")

    print("\ninternal validation set evaluation")
    val_metrics_dict = evaluate_model_sklearn(trained_model, x_val, y_val, scaler_instance=scaler)

    wandb.log(val_metrics_dict, step=num_subjects_to_use)

    print(f"\nusing model version: {model_version_to_use}")
    s3_model_key = f"{model_version_to_use}/model.joblib"
    s3_scaler_key = f"{model_version_to_use}/scaler.joblib"
    active_pointer_key = "active_model_pointer.txt"
    local_model_path = "./model.joblib"
    local_scaler_path = "./scaler.joblib"

    joblib.dump(trained_model, local_model_path)
    joblib.dump(scaler, local_scaler_path)

    try:
        s3_client.upload_file(local_model_path, MINIO_MODELS_BUCKET, s3_model_key)
        s3_client.upload_file(local_scaler_path, MINIO_MODELS_BUCKET, s3_scaler_key)
        s3_client.put_object(Bucket=MINIO_MODELS_BUCKET, Key=active_pointer_key, Body=model_version_to_use.encode('utf-8'))

        model_artifact = wandb.Artifact(
            name=model_version_to_use,
            type="model",
            description=f"model trained on {num_subjects_to_use} subjects. validation auc: {val_metrics_dict.get('auc', 'n/a'):.4f}",
            metadata={"num_subjects": num_subjects_to_use, "val_metrics": val_metrics_dict}
        )
        model_artifact.add_file(local_model_path, name="model.joblib")
        model_artifact.add_file(local_scaler_path, name="scaler.joblib")
        wandb.log_artifact(model_artifact)
        print("  model artifacts uploaded and w&b artifact logged")
    except Exception as e:
        print(f"  error during minio or w&b artifact upload: {e}")
    finally:
        if os.path.exists(local_model_path): os.remove(local_model_path)
        if os.path.exists(local_scaler_path): os.remove(local_scaler_path)

    wandb.finish()
    print(f"finished training pipeline. active model version: {model_version_to_use}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train p300 model from minio, log to w&b.")
    parser.add_argument("--training_subjects_percentage", type=float, default=1.0,
                        help="percentage of available processed subjects to use for training (0.0 to 1.0). defaulted to 1.0 (all available)")
    parser.add_argument("--internal_val_split_ratio", type=float, default=0.8,
                        help="ratio for splitting the selected subjects' data into internal train/validation. defaulted to 0.8")
    parser.add_argument("--model_version", type=str, default=None,
                        help="optional: specific model version string. if none, a timestamped version is generated")
    args = parser.parse_args()

    main_training_pipeline(
        training_subjects_percentage=args.training_subjects_percentage,
        internal_val_split_ratio=args.internal_val_split_ratio,
        cli_model_version=args.model_version
    )
