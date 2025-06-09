## Domain description

This machine learning pipeline is designed to classify P300 event-related potentials from electroencephalography (EEG) signals. The system processes raw EEG data to extract features, trains a classification model, and provides an API for inference.

Raw EEG data, specifically from the visual P300 Brain-Computer Interface (BCI) paradigm, is initially provided in individual subject-specific Parquet files in the project root directory on local machine. This raw data contains 16 EEG channels and 1 stimulus channel, sampled at 512 Hz.

The pipeline processes this raw data through a series of steps to prepare it for machine learning. Preprocessing involves filtering the EEG signals, segmenting the continuous data into short epochs time-locked to visual stimuli, performing baseline correction, and applying basic artifact rejection. Features are then extracted from these cleaned epochs by downsampling the data to 100 Hz and concatenating the time-series values across channels into a single feature vector. This processed data, along with corresponding labels (Target or Non-Target), is stored in an object storage emulating S3 API, MinIO, ready for model training or inference.

The core machine learning model used for classification is Linear Discriminant Analysis (LDA), which takes the extracted feature vectors as input to predict the presence or absence of a P300 response.

## Service-based pipeline structure

This MLOps pipeline utilizes a microservice architecture, with each service performing a distinct role in the machine learning workflow. Docker Compose manages these services.

MinIO serves as the central object storage solution. It hosts several S3-compatible buckets:
*   `p300-landing-zone` receives new raw subject data for initial intake.
*   `p300-raw-data` stores ingested raw data that is ready for preprocessing.
*   `p300-processed-features` contains features and labels extracted during preprocessing.
*   `p300-mlflow-artifacts` acts as the primary artifact store for MLflow, holding all logged models, scalers, and other run-specific files.
The `minio-init` service executes once at startup to create these buckets.

An Apache Airflow DAG, defined in `airflow-compose.yaml`, orchestrates data ingestion and preprocessing. This DAG simulates raw data arrival into `p300-landing-zone`, transfers it to `p300-raw-data`, and then triggers the `preprocessing-job` for subjects not yet processed. The `preprocessing-job` consumes data from `p300-raw-data` and outputs features to `p300-processed-features`.

The `training-job` service reads data from `p300-processed-features` to train a classification model (LDA) and a corresponding data scaler. It logs experiment parameters, metrics, and the trained model artifacts (model and scaler) to an MLflow Tracking Server. The MLflow server uses a PostgreSQL database (`postgres-mlflow`) for metadata storage and MinIO (the `p300-mlflow-artifacts` bucket) for artifact persistence. Crucially, the `training-job` registers the trained model to the MLflow Model Registry and can assign an alias (like "challenger") to the new model version.

Two model serving services, `model-serving-champion` and `model-serving-challenger`, host models for inference. These FastAPI applications load their respective models from the MLflow Model Registry at startup, based on environment variables specifying a model alias ("champion" or "challenger"). They retrieve the model and scaler artifacts from MinIO via MLflow.

An Nginx service (`nginx-load-balancer`) acts as a reverse proxy and load balancer. It directs incoming inference requests to the appropriate serving instances: requests to `/predict-champion/` are distributed across replicas of `model-serving-champion`, while requests to `/predict-challenger/` are sent to the `model-serving-challenger` instance.

The `application-simulator` service simulates real-world usage by fetching processed data, sending inference requests to the Nginx endpoints (targeting either champion or challenger), and collecting performance metrics. It queries MLflow to identify the specific model version under test (based on the alias) and logs these metrics along with version details to Weights & Biases for external tracking and visualization.

## Pipeline demonstration

This section outlines the steps to execute a sample workflow, and showcase the end-to-end functionality of the MLOps pipeline.

### 1. System initialization
We start all services defined in the main `docker-compose.yaml` and the Airflow-specific `airflow-compose.yaml`. This brings up MinIO, PostgreSQL, MLflow Server, Nginx, the model serving applications, and Airflow components.
   ```bash
   docker compose up --build -d
   docker compose -f airflow-compose.yaml up --build -d
   ```

### 2. Data ingestion and preprocessing
We interact with the Airflow DAG `p300_full_ingest_and_process_pipeline` to simulate new data arrival, ingest it, and preprocess it. This makes training data available in the `p300-processed-features` MinIO bucket. For this demonstration, we process data for 10 subjects by triggering the DAG twice.

First, we ensure the DAG is unpaused. Then, we trigger it. Finally, we can list the runs to check their status. These commands are executed inside the `airflow-webserver` container:
   ```bash
   docker compose -f airflow-compose.yaml exec airflow-webserver airflow dags unpause p300_full_ingest_and_process_pipeline
   docker compose -f airflow-compose.yaml exec airflow-webserver airflow dags trigger p300_full_ingest_and_process_pipeline
   # repeat the trigger command if more data is needed, for example, for 2 runs:
   docker compose -f airflow-compose.yaml exec airflow-webserver airflow dags trigger p300_full_ingest_and_process_pipeline 
   docker compose -f airflow-compose.yaml exec airflow-webserver airflow dags list-runs -d p300_full_ingest_and_process_pipeline
   ```
   *(Alternatively, these actions can be performed via the Airflow UI at `http://localhost:8080`)*

### 3. Deploying model Version 1 as Challenger
We execute the `training-job` to train a model using a subset of the processed data (50%, which is 5 subjects). The `--auto_set_challenger_alias` flag ensures the newly trained model (Version 1) is registered in MLflow and immediately assigned the "challenger" alias.
   ```bash
   docker compose run --rm --build training-job python train_model.py --training_subjects_percentage 0.5 --auto_set_challenger_alias
   ```
Following this, we restart the `model-serving-challenger` service. This prompts it to load the model version currently aliased as "challenger" (Version 1) from the MLflow Model Registry.
   ```bash
   docker compose restart model-serving-challenger
   ```

### 4. Evaluating Challenger model (Version 1)
We run the `application-simulator` from the local machine, configuring it to send inference requests to the challenger model's endpoint (via Nginx). This step evaluates the performance of the newly trained challenger model (Version 1) on a set of subjects (subjects 6 and 7). Metrics are logged to W&B.
   ```bash
   python application-simulator/application_simulator.py 6 7 --target_alias challenger --base_url http://localhost --mlflow_tracking_uri http://localhost:5001
   ```

### 5. Promoting Challenger (Version 1) to Champion
Assuming the challenger model (Version 1) performed satisfactorily, we promote it. In the MLflow UI (`http://localhost:5001`), we navigate to the "P300-Classifier" registered model, select Version 1, and change its alias from "challenger" to "champion". If no model previously held the "champion" alias, this makes Version 1 the active champion.
   Then, we restart the `model-serving-champion` replicas. They will now load Model Version 1 from the MLflow Registry as it holds the "champion" alias.
   ```bash
   docker compose restart model-serving-champion
   ```

### 6. Verifying the new Champion model (Version 1)
We run the `application-simulator` again, this time targeting the "champion" alias. This validates that the champion services are serving the newly promoted model (Version 1) using a different set of subjects (9 and 10).
   ```bash
   python application-simulator/application_simulator.py 9 10 --target_alias champion --base_url http://localhost --mlflow_tracking_uri http://localhost:5001
   ```

### 7. Deploying model Version 2 as new Challenger
We simulate further model development by training another model (Version 2), perhaps with more data (75%, 7 subjects). This new model is automatically assigned the "challenger" alias. Version 1 remains as "champion".
   ```bash
   docker compose run --rm --build training-job python train_model.py --training_subjects_percentage 0.75 --auto_set_challenger_alias
   ```
We restart the `model-serving-challenger` to load this new challenger (Version 2).
   ```bash
   docker compose restart model-serving-challenger
   ```

### 8. Evaluating the new Challenger model (Version 2)
We use the `application-simulator` to evaluate the performance of the new challenger model (Version 2).
   ```bash
   python application-simulator/application_simulator.py 9 10 --target_alias challenger --base_url http://localhost --mlflow_tracking_uri http://localhost:5001
   ```
 This completes a cycle of the ability to manage and deploy multiple model versions.

### 9. System teardown
To stop all services and remove associated containers, networks, and volumes for a clean state:
   ```bash
   docker compose -f airflow-compose.yaml down -v
   docker compose down -v
   ```
