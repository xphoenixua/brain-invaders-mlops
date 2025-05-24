FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir boto3 urllib3 pyarrow pandas

COPY init_minio.py .

ENTRYPOINT ["python", "init_minio.py"]
