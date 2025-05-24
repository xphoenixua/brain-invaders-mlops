import os
import argparse
import datetime
import io
import numpy as np
import pandas as pd
import requests
import wandb

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix


# constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_PROCESSED_BUCKET = os.getenv("MINIO_PROCESSED_BUCKET", "p300-processed-features")
SERVING_API_URL = os.getenv("SERVING_API_URL", "http://localhost:8000/predict/")
TARGET_CLASS_LABEL_CODE = 2 # original label code for 'target'


# s3 client initialization
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=BotoConfig(signature_version='s3v4', region_name='us-east-1'),
    use_ssl=False,
    verify=False
)


def download_processed_subject_data(subject_id, bucket_name):
    """downloads processed features and labels for a single subject from minio"""
    print(f"  downloading processed data for subject {subject_id:02d} from s3://{bucket_name}")
    features_key = f"subject_{subject_id:02d}/features.parquet"
    labels_key = f"subject_{subject_id:02d}/labels.parquet"

    try:
        features_obj = s3_client.get_object(Bucket=bucket_name, Key=features_key)
        x_features_df = pd.read_parquet(io.BytesIO(features_obj['Body'].read()))
        x_features = x_features_df.values

        labels_obj = s3_client.get_object(Bucket=bucket_name, Key=labels_key)
        y_labels_df = pd.read_parquet(io.BytesIO(labels_obj['Body'].read()))
        y_true_labels = y_labels_df['label'].values

        print(f"  downloaded data shapes: features={x_features.shape}, labels={y_true_labels.shape}")
        if x_features.shape[0] != y_true_labels.shape[0]:
            print(f"  warning: mismatch in number of features ({x_features.shape[0]}) and labels ({y_true_labels.shape[0]}) for subject {subject_id}")
        return x_features, y_true_labels
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"    warning: processed data not found for subject {subject_id:02d} in {bucket_name} (features: {features_key} or labels: {labels_key})")
        else:
            print(f"    error downloading data for subject {subject_id:02d}: {e}")
        return None, None
    except Exception as e:
        print(f"    unexpected error downloading for subject {subject_id:02d}: {e}")
        return None, None


def simulate_batch_predictions(subject_ids_for_inference, batch_step_id=None):
    """simulates predictions for a batch of subjects, logs per-subject and aggregated metrics to w&b"""
    batch_run_name = f"inference_batch_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    if batch_step_id is not None: batch_run_name += f"_step{batch_step_id}"

    wandb_run = None
    try:
        wandb.init(project="p300-bci-mlops", name=batch_run_name, job_type="batch_inference", reinit=True)
        wandb.config.subject_ids_inferred_in_batch = subject_ids_for_inference
        if batch_step_id is not None: wandb.config.batch_step_id = batch_step_id
        wandb_run = wandb.run
    except Exception as e:
        print(f"W&B init failed: {e}. proceeding without W&B logging for this batch")

    print(f"simulating predictions for batch of subjects: {subject_ids_for_inference}")

    per_subject_metrics_for_table = []
    overall_true_binary_labels = []
    overall_predicted_binary_labels = []
    overall_target_probabilities = []
    overall_true_binary_for_auc = []

    active_model_version_from_api = "unknown"

    for subject_id_idx, subject_id in enumerate(subject_ids_for_inference):
        print(f"\n  processing subject {subject_id:02d}")
        x_features, y_true_original_labels = download_processed_subject_data(subject_id, MINIO_PROCESSED_BUCKET)

        if x_features is None or x_features.shape[0] == 0:
            print(f"    no data or empty features for subject {subject_id:02d}. skipping")
            per_subject_metrics_for_table.append([subject_id, None, None, None, None, None, None, None])
            continue

        y_pred_api_label_codes_subject = []
        y_target_probas_api_subject = []

        for i in range(x_features.shape[0]):
            feature_vector = x_features[i, :].tolist()
            payload = {"features": feature_vector}
            try:
                response = requests.post(SERVING_API_URL, json=payload)
                response.raise_for_status()
                prediction_result = response.json()

                y_pred_api_label_codes_subject.append(prediction_result.get("predicted_label_code"))
                y_target_probas_api_subject.append(prediction_result.get("probability_target"))

                if subject_id_idx == 0 and i == 0 and "model_version" in prediction_result:
                    active_model_version_from_api = prediction_result.get("model_version")
                    if wandb_run: wandb.config.model_version_used_for_batch = active_model_version_from_api
            except requests.exceptions.RequestException as e:
                print(f"    error sending prediction request for epoch {i} of subject {subject_id}: {e}")
                y_pred_api_label_codes_subject.append(None)
                y_target_probas_api_subject.append(None)
            except Exception as e:
                print(f"    unexpected error processing epoch {i} of subject {subject_id} prediction: {e}")
                y_pred_api_label_codes_subject.append(None)
                y_target_probas_api_subject.append(None)

        # per-subject metrics calculation
        pred_codes_arr = np.array(y_pred_api_label_codes_subject, dtype=np.float64)
        probas_arr = np.array(y_target_probas_api_subject, dtype=np.float64)

        valid_pred_indices = ~np.isnan(pred_codes_arr)
        y_true_orig_for_pred_metrics = y_true_original_labels[valid_pred_indices]
        y_pred_api_codes_for_pred_metrics = pred_codes_arr[valid_pred_indices].astype(int)

        y_true_subj_binary = np.where(y_true_orig_for_pred_metrics == TARGET_CLASS_LABEL_CODE, 1, 0)
        y_pred_subj_binary = np.where(y_pred_api_codes_for_pred_metrics == TARGET_CLASS_LABEL_CODE, 1, 0)

        overall_true_binary_labels.extend(y_true_subj_binary)
        overall_predicted_binary_labels.extend(y_pred_subj_binary)

        valid_proba_indices = ~np.isnan(probas_arr)
        y_true_orig_for_auc = y_true_original_labels[valid_proba_indices]
        y_true_subj_binary_for_auc = np.where(y_true_orig_for_auc == TARGET_CLASS_LABEL_CODE, 1, 0)
        y_proba_subj_for_auc = probas_arr[valid_proba_indices]

        overall_true_binary_for_auc.extend(y_true_subj_binary_for_auc)
        overall_target_probabilities.extend(y_proba_subj_for_auc)

        subj_acc, subj_kap, subj_auc = None, None, None
        subj_cm = [None, None, None, None]

        if len(y_true_subj_binary) > 0:
            subj_acc = accuracy_score(y_true_subj_binary, y_pred_subj_binary)
            subj_kap = cohen_kappa_score(y_true_subj_binary, y_pred_subj_binary)
            cm_raw = confusion_matrix(y_true_subj_binary, y_pred_subj_binary)
            if cm_raw.size == 4: subj_cm = cm_raw.ravel().tolist()

            if len(y_true_subj_binary_for_auc) > 0 and len(np.unique(y_true_subj_binary_for_auc)) > 1:
                try:
                    subj_auc = roc_auc_score(y_true_subj_binary_for_auc, y_proba_subj_for_auc)
                except Exception as e_auc:
                    print(f"    warning: AUC calculation failed for subject {subject_id}: {e_auc}")

        per_subject_metrics_for_table.append([
            subject_id, subj_acc, subj_kap, subj_auc,
            subj_cm[0], subj_cm[1], subj_cm[2], subj_cm[3]
        ])
        print(
            f"    s{subject_id:02d} metrics: "
            f"acc={f'{subj_acc:.3f}' if subj_acc is not None else 'n/a'}, "
            f"kap={f'{subj_kap:.3f}' if subj_kap is not None else 'n/a'}, "
            f"auc={f'{subj_auc:.3f}' if subj_auc is not None else 'n/a'}"
        )

    # log per-subject metrics as a w&b table
    if wandb_run and per_subject_metrics_for_table:
        columns = ["subjectid", "accuracy", "kappa", "auc", "tn", "fp", "fn", "tp"]
        subject_table = wandb.Table(columns=columns, data=per_subject_metrics_for_table)
        wandb.log({"per_subject_inference_metrics_table": subject_table}, step=batch_step_id)

    # aggregated metrics for the batch
    agg_metrics_to_log = {}
    if overall_true_binary_labels:
        print("\naggregated batch metrics")
        agg_accuracy = accuracy_score(overall_true_binary_labels, overall_predicted_binary_labels)
        agg_kappa = cohen_kappa_score(overall_true_binary_labels, overall_predicted_binary_labels)
        print(f"  overall accuracy: {agg_accuracy:.4f}")
        print(f"  overall kappa: {agg_kappa:.4f}")
        agg_metrics_to_log["batch_accuracy"] = agg_accuracy
        agg_metrics_to_log["batch_kappa"] = agg_kappa

        if overall_true_binary_for_auc and len(np.unique(overall_true_binary_for_auc)) > 1:
            try:
                agg_auc = roc_auc_score(overall_true_binary_for_auc, overall_target_probabilities)
                print(f"  overall auc: {agg_auc:.4f}")
                agg_metrics_to_log["batch_auc"] = agg_auc
            except Exception as e:
                print(f"  could not calculate aggregated auc: {e}")
        else:
            print("  could not calculate aggregated AUC, insufficient data or only one class")

        if wandb_run: wandb.log(agg_metrics_to_log, step=batch_step_id if batch_step_id is not None else wandb.run.step)

    if wandb_run: wandb.finish()
    print(f"finished batch simulation for subjects: {subject_ids_for_inference}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulate app for batch of subjects, log to W&B.")
    parser.add_argument("subject_ids", nargs='+', type=int,
                        help="list of subject ids to simulate, like 6 7 8.")
    parser.add_argument("--batch_step", type=int, default=None,
                        help="optional step id for this batch for W&B logging, like number of batches processed so far.")
    args = parser.parse_args()
    simulate_batch_predictions(args.subject_ids, batch_step_id=args.batch_step)
