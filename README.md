# Pipeline on Databricks
This branch contains the implementation of the P300 classification MLOps pipeline on the Databricks platform. It adapts the core logic from the `master` branch's Docker-based services into a series of Databricks notebooks and leverages managed Databricks services for orchestration, experiment tracking, and model serving.
## Overview
This implementation replaces the self-hosted Docker components with their Databricks-managed equivalents:
* Databricks Workflows (Jobs) replace Airflow.
* Databricks File System (DBFS) replaces MinIO.
* Managed MLflow on Databricks replaces the self-hosted MLflow server and PostgreSQL backend.
* Databricks Model Serving replaces the custom FastAPI/Nginx deployment, providing a managed endpoint for the champion/challenger strategy.
## Notebooks workflow
The pipeline is organized into a series of notebooks, designed to be run as tasks within Databricks Jobs.
1.  **`00_ingest_data`** simulates new data arrival by moving raw subject files from a landing zone directory to a raw data directory on DBFS.
2.  **`01_orchestrate_preprocessing`** identifies new subjects in the raw data directory and triggers the `01a_process_single_subject` notebook for each one in a batch.
3.  **`01a_process_single_subject`** is the core worker notebook that preprocesses a single subject's raw EEG data and saves the resulting features to DBFS.
4.  **`02_train_model`** reads all processed features, trains an LDA model and scaler, logs the experiment to MLflow, and registers the packaged model to the Model Registry, assigning it to the `Staging` stage.
5.  **`03_deploy_model`** is an idempotent deployment script. It checks if a serving endpoint exists. If not, it creates one. It then updates the endpoint to serve the latest model in the `Staging` stage as the challenger.
6.  **`04_run_simulation`** evaluates a deployed model (champion or challenger). It identifies the model version being served, creates a holdout set by excluding subjects the model was trained on, and sends batched inference requests to the model's direct invocation URL. It logs the final performance metrics to MLflow.
7.  **`05_promote_challenger_to_champion`** automates the promotion process. It identifies the current challenger model, transitions it to the `Production` stage in the registry, and updates the serving endpoint to serve this new version as the champion.
8.  
## How to run
### Setup
*   Create an All-Purpose cluster using a Databricks ML Runtime.
*   Upload the raw subject data (`subject_XX_eeg_stim.parquet` files) to `/dbfs/FileStore/tables/p300_files/landing-zone/`.
*   Import all the notebooks into your Databricks workspace.
### Create jobs
* **Job 1: `Ingest_and_Preprocess`** – a job with two sequential tasks running `00_ingest_data` and then `01_orchestrate_preprocessing`.
* **Job 2: `Train_and_Deploy_Challenger`** – a job with two sequential tasks running `02_train_model` and then `03_deploy_model`.
* **Job 3: `Evaluate_Model`** – a job running the `04_run_simulation` notebook. Configure its `target_entity_name` parameter to either "champion" or "challenger".
* **Job 4: `Promote_to_Champion`** – a job running the `05_promote_challenger_to_champion` notebook.
### Execution order
1. Run the `Ingest_and_Preprocess` job to prepare data.
2. Run the `Train_and_Deploy_Challenger` job to train a model and deploy it.
3. Run the `Evaluate_Model` job (targeting the challenger) to test its performance.
4. If satisfied, run the `Promote_to_Champion` job.
5. Repeat the train/evaluate/promote cycle as needed.
