import os
import time
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig


# constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
LANDING_ZONE_BUCKET_NAME = os.getenv("LANDING_ZONE_BUCKET_NAME")
RAW_BUCKET_NAME = os.getenv("RAW_BUCKET_NAME")
PROCESSED_BUCKET_NAME = os.getenv("PROCESSED_BUCKET_NAME")
MODELS_BUCKET_NAME = os.getenv("MODELS_BUCKET_NAME")
MLFLOW_ARTIFACT_BUCKET_NAME = os.getenv("MLFLOW_ARTIFACT_BUCKET_NAME")

MINIO_WAIT_RETRIES = 20
MINIO_WAIT_DELAY_SECONDS = 5


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


def wait_for_minio(endpoint, access_key, secret_key):
    """waits for minio server to become ready"""
    print(f"waiting for minio at {endpoint}")
    # specific client for health check with short timeouts
    s3_client_healthcheck = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version='s3v4', region_name='us-east-1', connect_timeout=1, read_timeout=1),
        use_ssl=False,
        verify=False
    )

    for i in range(1, MINIO_WAIT_RETRIES + 1):
        try:
            s3_client_healthcheck.list_buckets()
            print("minio is ready")
            return True
        except ClientError as e:
            print(f"attempt {i}/{MINIO_WAIT_RETRIES}: minio client error: {e}. waiting {MINIO_WAIT_DELAY_SECONDS} seconds")
        except Exception as e: # catch connection errors (like urllib3.exceptions.NewConnectionError)
            print(f"attempt {i}/{MINIO_WAIT_RETRIES}: minio connection error: {e}. waiting {MINIO_WAIT_DELAY_SECONDS} seconds")
        time.sleep(MINIO_WAIT_DELAY_SECONDS)

    print("minio did not become ready")
    return False


def create_bucket_if_not_exists(s3_client, bucket_name):
    """ensures an s3 bucket exists, creating it if necessary"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"bucket '{bucket_name}' already exists")
        return
    except ClientError as e:
        if e.response['Error']['Code'] != '404':
            raise RuntimeError(f"error checking existence of bucket '{bucket_name}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"unexpected error checking existence of bucket '{bucket_name}': {e}") from e

    try:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"bucket '{bucket_name}' created")
    except ClientError as e:
        raise RuntimeError(f"error creating bucket '{bucket_name}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"unexpected error creating bucket '{bucket_name}': {e}") from e


def initialize_minio_buckets():
    """initializes minio by waiting for service and creating necessary buckets"""
    if not wait_for_minio(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY):
        exit(1)

    s3 = create_s3_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)

    buckets_to_create = [
        LANDING_ZONE_BUCKET_NAME,
        RAW_BUCKET_NAME,
        PROCESSED_BUCKET_NAME,
        MODELS_BUCKET_NAME,
        MLFLOW_ARTIFACT_BUCKET_NAME
    ]

    for bucket_name in buckets_to_create:
        if not bucket_name: continue # skip if bucket name is empty string from env
        try:
            create_bucket_if_not_exists(s3, bucket_name)
        except RuntimeError as e:
            print(f"critical error during bucket initialization: {e}")
            exit(1)

    print("\nminio initialization complete! all buckets created and start empty")


if __name__ == "__main__":
    initialize_minio_buckets()
