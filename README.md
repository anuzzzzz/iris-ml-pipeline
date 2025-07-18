
# IRIS ML Pipeline - DVC Setup Complete

## Project Structure:
- Raw data tracked with DVC
- Processed data pipeline
- Model training pipeline
- DVC remote storage in GCS
- Basic testing framework

## DVC Features Implemented:
✅ Data versioning
✅ Remote storage (GCS)
✅ Pipeline automation
✅ Metrics tracking
✅ Reproducible workflows

## Next Steps:
- Setup Feature Store (Feast)
- Integrate MLflow for experiment tracking
- Setup GitHub repository with CI/CD
- Add hyperparameter tuning
- Deploy to Vertex AI

## Commands to Remember:
- `dvc repro` - Run pipeline
- `dvc push` - Push to remote
- `dvc pull` - Pull from remote
- `dvc metrics show` - Show metrics
- `dvc dag` - Show pipeline DAG

