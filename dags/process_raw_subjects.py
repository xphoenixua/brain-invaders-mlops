from __future__ import annotations

import pendulum
import logging
import os

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.decorators import task
from airflow.models import Variable


# constants
MINIO_ENDPOINT = "http://minio-server:9000" # !! should be environment variable in later versions
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

LANDING_ZONE_BUCKET = "p300-landing-zone"
RAW_DATA_BUCKET = "p300-raw-data"
PROCESSED_FEATURES_BUCKET = "p300-processed-features"
MODELS_BUCKET = "p300-models"

DOCKER_NETWORK = "braininvaders_mlops_default"
PREPROCESSING_IMAGE_NAME = "preprocessing-service:latest"

ARRIVAL_BATCH_SIZE = 5 # number of subjects to arrive in landing zone per DAG run
INGESTION_BATCH_SIZE = 5 # number of subjects to ingest from landing to raw per run
PREPROCESSING_BATCH_SIZE = 5 # number of subjects to preprocess from raw to processed per run

LAST_ARRIVED_SUBJECT_ID_VAR = "last_arrived_subject_id" # Airflow variable to track last arrived subject ID
LOCAL_ALL_RAW_SUBJECTS_PATH = "/opt/airflow/all_raw_subjects_local"


def get_s3_client():
    """returns a configured boto3 s3 client for minio"""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        use_ssl=False,
        verify=False,
        config=BotoConfig(signature_version='s3v4', region_name='us-east-1')
    )


# @task decorator is syntactic sugar provided by Airflow
# internally, it converts the function into a PythonOperator, which is the base operator for executing Python code
# so we don't need to explicitly use PythonOperator in the DAG definition
# for which we would have still needed to define functions like these below but without the @task decorator
@task
def raw_data_arrival() -> list[int]:
    """
    simulates new raw subject data arriving by copying files from a local source
    into the MinIO landing zone bucket. it updates an Airflow variable for the next id
    """
    logger = logging.getLogger("airflow.task")
    s3 = get_s3_client()

    last_id_str = Variable.get(LAST_ARRIVED_SUBJECT_ID_VAR, default_var="0", deserialize_json=False)
    next_sub_id = int(last_id_str) + 1

    arrived_sub_ids = []

    for i in range(ARRIVAL_BATCH_SIZE):
        subject_id = next_sub_id + i
        subject_folder_name = f"subject_{subject_id:02d}"
        local_sub_path = os.path.join(LOCAL_ALL_RAW_SUBJECTS_PATH, subject_folder_name)

        if not os.path.isdir(local_sub_path):
            logger.info(f"local raw data for subject {subject_id} not found at {local_sub_path}. stopping data arrival.")
            break

        logger.info(f"arriving subject {subject_id} from {local_sub_path} to landing zone.")
        try:
            for root, _, files in os.walk(local_sub_path):
                for file_name in files:
                    local_file_path = os.path.join(root, file_name)
                    s3_object_key = f"{subject_folder_name}/{file_name}"
                    s3.upload_file(local_file_path, LANDING_ZONE_BUCKET, s3_object_key)
            logger.info(f"  subject {subject_id} successfully landed in {LANDING_ZONE_BUCKET}.")
            arrived_sub_ids.append(subject_id)
        except Exception as e:
            logger.error(f"error arriving data for subject {subject_id}: {e}.")

    if arrived_sub_ids:
        new_last_id = max(arrived_sub_ids)
        Variable.set(LAST_ARRIVED_SUBJECT_ID_VAR, str(new_last_id))
        logger.info(f"updated {LAST_ARRIVED_SUBJECT_ID_VAR} to {new_last_id}.")
        logger.info(f"arrived subjects: {arrived_sub_ids}.")
    else:
        logger.info("no subjects arrived in this run.")

    return arrived_sub_ids


@task
def ingest_subjects_from_landing_zone(arrived_subject_ids: list[int]) -> list[int]:
    """
    ingests subjects from the landing zone bucket to the raw data bucket.
    it depends on the output of the raw_data_arrival task
    """
    logger = logging.getLogger("airflow.task")
    s3 = get_s3_client()

    subs_in_landing = set()
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=LANDING_ZONE_BUCKET, Delimiter='/'):
            for prefix_obj in page.get('CommonPrefixes', []):
                folder_name = prefix_obj['Prefix'].strip('/')
                if folder_name.startswith("subject_"):
                    try: subs_in_landing.add(int(folder_name.split('_')[1]))
                    except (ValueError, IndexError): logger.warning(f"skipping malformed folder in landing zone: {folder_name}.")
    except Exception as e:
        logger.error(f"failed to list objects from landing zone: {e}.")
        return []

    subs_to_ingest = sorted(list(subs_in_landing))[:INGESTION_BATCH_SIZE]
    ingested_sub_ids = []

    if not subs_to_ingest:
        logger.info("no subjects found in landing zone to ingest for this batch.")
        return []

    logger.info(f"attempting to ingest subjects: {subs_to_ingest} from landing zone to raw data bucket.")
    for subject_id in subs_to_ingest:
        landing_prefix = f"subject_{subject_id:02d}/"

        files_to_delete_from_landing = []
        try:
            objects_to_copy = s3.list_objects_v2(Bucket=LANDING_ZONE_BUCKET, Prefix=landing_prefix)
            if not objects_to_copy.get('Contents'):
                logger.warning(f"  no files found for subject {subject_id} in landing zone prefix {landing_prefix}. skipping.")
                continue

            for obj in objects_to_copy['Contents']:
                src_key = obj['Key']
                target_key = src_key # target key is same structure as source key
                copy_src = {'Bucket': LANDING_ZONE_BUCKET, 'Key': src_key}
                logger.info(f"  copying {src_key} to s3://{RAW_DATA_BUCKET}/{target_key}.")
                s3.copy_object(CopySource=copy_src, Bucket=RAW_DATA_BUCKET, Key=target_key)
                files_to_delete_from_landing.append({'Key': src_key})

            if files_to_delete_from_landing:
                logger.info(f"  successfully copied subject {subject_id}. deleting from landing zone.")
                delete_payload = {'Objects': files_to_delete_from_landing}
                s3.delete_objects(Bucket=LANDING_ZONE_BUCKET, Delete=delete_payload)
                logger.info(f"  deleted subject {subject_id} data from landing zone.")
                ingested_sub_ids.append(subject_id)

        except Exception as e:
            logger.error(f"  error during ingestion (copy or delete) for subject {subject_id}: {e}.")

    if ingested_sub_ids:
        logger.info(f"successfully ingested and cleared from landing: {ingested_sub_ids}.")
    else:
        logger.info("no subjects were successfully ingested and cleared from landing in this run.")

    return ingested_sub_ids


@task
def identify_subjects_for_preprocessing(ingested_subject_ids: list[int]) -> list[dict]:
    """
    compares subjects in the raw data bucket against the processed features bucket
    and returns a batch of subject ids that are in raw but not yet processed.
    returns a list of dictionaries for expand_kwargs
    """
    logger = logging.getLogger("airflow.task")
    s3 = get_s3_client()

    def get_subject_ids_from_bucket(bucket: str) -> set[int]:
        try:
            paginator = s3.get_paginator('list_objects_v2')
            response_iterator = paginator.paginate(Bucket=bucket, Delimiter='/')
            sub_ids = set()
            for page in response_iterator:
                for prefix_obj in page.get('CommonPrefixes', []):
                    folder_name = prefix_obj['Prefix'].strip('/')
                    if folder_name.startswith("subject_"):
                        try: sub_ids.add(int(folder_name.split('_')[1]))
                        except (ValueError, IndexError): logger.warning(f"skipping malformed folder in {bucket}: {folder_name}.")
            return sub_ids
        except Exception as e:
            logger.error(f"failed to list objects from bucket '{bucket}': {e}.")
            return set()

    all_subs_in_raw = get_subject_ids_from_bucket(RAW_DATA_BUCKET)
    processed_subs = get_subject_ids_from_bucket(PROCESSED_FEATURES_BUCKET)

    unprocessed_subs_in_raw = sorted(list(all_subs_in_raw - processed_subs))

    subs_to_process = unprocessed_subs_in_raw[:PREPROCESSING_BATCH_SIZE]

    if subs_to_process:
        logger.info(f"identified {len(subs_to_process)} subjects for preprocessing: {subs_to_process}.")
    else:
        logger.info("no subjects in raw data bucket ready for preprocessing.")

    return [{"command": ["python", "preprocess_subject.py", str(subject_id)]} for subject_id in subs_to_process]


with DAG(
    dag_id='p300_full_ingest_and_process_pipeline',
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=['p300'],
) as dag:

    start = EmptyOperator(task_id="start_pipeline_run")

    # task 1: raw data arrival
    arrived_sub_ids_xcom = raw_data_arrival()

    # task 2: ingest subjects from landing zone to raw data bucket
    ingested_sub_ids_xcom = ingest_subjects_from_landing_zone(
        arrived_subject_ids=arrived_sub_ids_xcom
    )

    # task 3: identify which subjects (from those just ingested or already in raw) need preprocessing
    preprocess_kwargs = identify_subjects_for_preprocessing(
        ingested_subject_ids=ingested_sub_ids_xcom
    )

    # task 4: preprocess each identified subject using docker operator
    preprocess_subjects_task = DockerOperator.partial(
        task_id="preprocess_subject_batch",
        image=PREPROCESSING_IMAGE_NAME,
        api_version="auto",
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove='success',
        environment={
            "MINIO_ENDPOINT": MINIO_ENDPOINT,
            "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
            "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
            "MINIO_RAW_BUCKET": RAW_DATA_BUCKET,
            "MINIO_PROCESSED_BUCKET": PROCESSED_FEATURES_BUCKET,
        },
    ).expand_kwargs(preprocess_kwargs)

    end = EmptyOperator(task_id="end_pipeline_run", trigger_rule="all_done")

    start >> arrived_sub_ids_xcom >> ingested_sub_ids_xcom >> preprocess_kwargs >> preprocess_subjects_task >> end
