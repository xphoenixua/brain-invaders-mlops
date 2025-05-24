import os
import argparse
import io
import numpy as np
import pandas as pd
import mne

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig


# constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio-server:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_RAW_BUCKET = os.getenv("MINIO_RAW_BUCKET")
MINIO_PROCESSED_BUCKET = os.getenv("MINIO_PROCESSED_BUCKET")

PREPROCESSING_CONFIG = {
    "sfreq": 512,
    "filter_l_freq": 1.0,
    "filter_h_freq": 20.0,
    "notch_freqs": np.arange(50, 251, 50).tolist(),
    "epoch_tmin": -0.1,
    "epoch_tmax": 0.6,
    "baseline_correction": (-0.1, 0.0),
    "reject_criteria_eeg": 100e-6,
    "event_id": {'NonTarget': 1, 'Target': 2},
    "stim_channel_name": "stim"
}
FEATURE_EXTRACTION_CONFIG = {
    "resample_sfreq": 100
}


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


def create_mne_raw_from_df(df, sfreq, stim_channel_name="stim"):
    """converts a pandas dataframe to an mne raw object"""
    eeg_channels = [col for col in df.columns if col.startswith('eeg_ch')]
    all_channels = eeg_channels + [stim_channel_name]
    eeg_types = ['eeg'] * len(eeg_channels)
    all_types = eeg_types + ['stim']

    eeg_data = df[eeg_channels].T.values * 1e-6 # convert to volts
    stim_data = df[stim_channel_name].values.astype(np.int64)[None, :]
    data_for_mne = np.concatenate([eeg_data, stim_data], axis=0)

    info = mne.create_info(ch_names=all_channels, sfreq=sfreq, ch_types=all_types, verbose=False)
    raw = mne.io.RawArray(data_for_mne, info, verbose=False)
    return raw


def preprocess_raw_eeg(mne_raw_object, config):
    """applies filtering and finds events on an MNE raw object"""
    print(f"    filtering raw data ({config['filter_l_freq']}-{config['filter_h_freq']} hz)")
    mne_raw_object.filter(config['filter_l_freq'], config['filter_h_freq'], fir_design='firwin', verbose=False)

    if config.get('notch_freqs'):
        # filter out notch freqs outside the bandpass range to avoid mne warnings
        valid_notch_freqs = [f for f in config['notch_freqs'] if config['filter_l_freq'] < f < config['filter_h_freq']]
        if valid_notch_freqs: mne_raw_object.notch_filter(freqs=valid_notch_freqs, fir_design='firwin', verbose=False)

    events = mne.find_events(mne_raw_object, stim_channel=config['stim_channel_name'], shortest_event=1, verbose=False)
    return mne_raw_object, events


def epoch_data(mne_raw_object, events, config):
    """epochs the mne raw data based on events"""
    print(f"    epoching data ({config['epoch_tmin']}s to {config['epoch_tmax']}s)")
    epochs = mne.Epochs(mne_raw_object, events, config['event_id'],
                        tmin=config['epoch_tmin'], tmax=config['epoch_tmax'],
                        baseline=config['baseline_correction'], picks='eeg',
                        preload=True, reject_by_annotation=False, verbose=False)

    if config.get('reject_criteria_eeg'):
        epochs.drop_bad(reject={'eeg': config['reject_criteria_eeg']}, verbose=False)
    return epochs


def extract_features(mne_epochs_object, config):
    """extracts flattened features and labels from mne epochs"""
    if not len(mne_epochs_object): return np.array([]), np.array([])
    
    epochs_resampled = mne_epochs_object.copy().resample(sfreq=config['resample_sfreq'], npad='auto', verbose=False)
    x_data = epochs_resampled.get_data()
    y_labels = epochs_resampled.events[:, -1] # last column contains event ids

    n_epochs, n_channels, n_times_resampled = x_data.shape
    x_flattened = x_data.reshape(n_epochs, n_channels * n_times_resampled)
    return x_flattened, y_labels


def process_subject(subject_id):
    """orchestrates the preprocessing pipeline for a single subject"""
    print(f"starting preprocessing for subject {subject_id:02d}")

    # download raw data
    raw_data_key = f"subject_{subject_id:02d}/subject_{subject_id:02d}_eeg_stim.parquet"
    try:
        print(f"  downloading raw data from s3://{MINIO_RAW_BUCKET}/{raw_data_key}")
        response = s3_client.get_object(Bucket=MINIO_RAW_BUCKET, Key=raw_data_key)
        df_subject_raw = pd.read_parquet(io.BytesIO(response['Body'].read()))
        print(f"  raw data for subject {subject_id:02d} loaded. shape: {df_subject_raw.shape}")
    except Exception as e:
        print(f"  error downloading or reading raw data for subject {subject_id:02d} from minio: {e}")
        return

    # mne preprocessing steps
    raw_mne = create_mne_raw_from_df(df_subject_raw, PREPROCESSING_CONFIG['sfreq'], PREPROCESSING_CONFIG['stim_channel_name'])
    raw_mne_processed, events = preprocess_raw_eeg(raw_mne, PREPROCESSING_CONFIG)
    
    if not events or len(events) == 0:
        print(f"  no events found for subject {subject_id:02d} after initial processing. skipping")
        return

    epochs_subject = epoch_data(raw_mne_processed, events, PREPROCESSING_CONFIG)
    if not epochs_subject or len(epochs_subject) == 0:
        print(f"  no epochs remaining for subject {subject_id:02d} after artifact rejection. skipping")
        return

    x_subject_features, y_subject_labels = extract_features(epochs_subject, FEATURE_EXTRACTION_CONFIG)
    if x_subject_features.size == 0:
        print(f"  no features extracted for subject {subject_id:02d}. skipping")
        return
    print(f"  extracted features shape: {x_subject_features.shape}, labels shape: {y_subject_labels.shape}")

    # upload processed features and labels to minio
    df_features = pd.DataFrame(x_subject_features)
    df_labels = pd.DataFrame(y_subject_labels, columns=['label'])

    processed_features_key = f"subject_{subject_id:02d}/features.parquet"
    processed_labels_key = f"subject_{subject_id:02d}/labels.parquet"

    try:
        features_buffer = io.BytesIO()
        df_features.to_parquet(features_buffer, index=False)
        features_buffer.seek(0)
        s3_client.put_object(Bucket=MINIO_PROCESSED_BUCKET, Key=processed_features_key, Body=features_buffer)
        print(f"  uploaded processed features to s3://{MINIO_PROCESSED_BUCKET}/{processed_features_key}")

        labels_buffer = io.BytesIO()
        df_labels.to_parquet(labels_buffer, index=False)
        labels_buffer.seek(0)
        s3_client.put_object(Bucket=MINIO_PROCESSED_BUCKET, Key=processed_labels_key, Body=labels_buffer)
        print(f"  uploaded processed labels to s3://{MINIO_PROCESSED_BUCKET}/{processed_labels_key}")

    except Exception as e:
        print(f"  error uploading processed data for subject {subject_id:02d} to minio: {e}")

    print(f"finished preprocessing for subject {subject_id:02d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess eeg data for a single subject from minio.")
    parser.add_argument("subject_id", type=int, help="id of the subject to process, like 1 or 2.")
    args = parser.parse_args()

    process_subject(args.subject_id)
