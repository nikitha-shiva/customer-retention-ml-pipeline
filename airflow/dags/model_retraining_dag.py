"""
Airflow DAG for Automated Customer Retention Model Retraining
Implements drift detection, automated retraining, and model deployment
Author: Nikitha Shiva
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DAG Configuration
DAG_ID = "customer_retention_model_retraining"
SCHEDULE_INTERVAL = "@weekly"  # Weekly retraining
MAX_ACTIVE_RUNS = 1

# Default arguments
default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['data-team@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Configuration from Airflow Variables
PERFORMANCE_THRESHOLD = float(Variable.get("retention_model_performance_threshold", 0.85))
DRIFT_THRESHOLD = float(Variable.get("retention_model_drift_threshold", 0.05))
MODEL_PATH = Variable.get("retention_model_path", "/models/customer_retention")
DATA_PATH = Variable.get("retention_data_path", "/data/customer_retention")

def extract_recent_data(**context) -> Dict[str, Any]:
    """
    Extract recent customer data for model evaluation and retraining
    
    Returns:
        Dictionary with data extraction results
    """
    logger.info("Starting data extraction for model retraining...")
    
    try:
        # This would typically connect to your data warehouse
        # For demo, we'll simulate the process
        
        # Extract customer data from the last 30 days
        extraction_query = """
        SELECT 
            customer_id,
            recency_30d,
            frequency_30d,
            monetary_total_30d,
            engagement_score,
            customer_tenure_months,
            support_tickets_30d,
            churned,
            created_at
        FROM customer_features 
        WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
        AND churned IS NOT NULL
        """
        
        # In real implementation, execute against your data warehouse
        # df = pd.read_sql(extraction_query, connection)
        
        # Simulate data extraction results
        extraction_results = {
            'records_extracted': 10000,
            'extraction_date': datetime.now().isoformat(),
            'data_quality_score': 0.95,
            'missing_values_pct': 2.5,
            'extraction_time_seconds': 45
        }
        
        # Store extraction metadata
        context['task_instance'].xcom_push(
            key='extraction_results', 
            value=extraction_results
        )
        
        logger.info(f"Data extraction completed: {extraction_results['records_extracted']} records")
        return extraction_results
        
    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        raise

def detect_model_drift(**context) -> Dict[str, Any]:
    """
    Detect model performance drift and data distribution changes
    
    Returns:
        Dictionary with drift detection results
    """
    logger.info("Starting model drift detection...")
    
    try:
        # Get extraction results from previous task
        extraction_results = context['task_instance'].xcom_pull(
            task_ids='extract_recent_data',
            key='extraction_results'
        )
        
        # Load current production model
        try:
            current_model = joblib.load(f"{MODEL_PATH}/customer_retention_model.pkl")
            model_metadata = joblib.load(f"{MODEL_PATH}/model_metadata.json")
        except FileNotFoundError:
            logger.warning("No existing model found, will train new model")
            drift_results = {
                'performance_drift': 0.0,
                'data_drift_detected': False,
                'requires_retraining': True,
                'current_accuracy': 0.0,
                'baseline_accuracy': 0.0,
                'drift_score': 0.0
            }
            context['task_instance'].xcom_push(key='drift_results', value=drift_results)
            return drift_results
        
        # Simulate performance evaluation on recent data
        # In real implementation, you would:
        # 1. Load recent labeled data
        # 2. Make predictions with current model
        # 3. Calculate performance metrics
        # 4. Compare with baseline performance
        
        # Simulated current performance
        current_accuracy = np.random.normal(0.88, 0.02)  # Simulate some variation
        baseline_accuracy = float(model_metadata.get('baseline_accuracy', 0.90))
        
        # Calculate performance drift
        performance_drift = (baseline_accuracy - current_accuracy) / baseline_accuracy
        
        # Simulate data drift detection
        # In real implementation, use statistical tests like KS test, PSI, etc.
        drift_score = np.random.normal(0.02, 0.01)  # Simulate drift score
        data_drift_detected = drift_score > DRIFT_THRESHOLD
        
        # Determine if retraining is required
        requires_retraining = (
            performance_drift > DRIFT_THRESHOLD or
            data_drift_detected or
            current_accuracy < PERFORMANCE_THRESHOLD
        )
        
        drift_results = {
            'performance_drift': round(performance_drift, 4),
            'data_drift_detected': data_drift_detected,
            'requires_retraining': requires_retraining,
            'current_accuracy': round(current_accuracy, 4),
            'baseline_accuracy': baseline_accuracy,
            'drift_score': round(drift_score, 4),
            'drift_threshold': DRIFT_THRESHOLD,
            'performance_threshold': PERFORMANCE_THRESHOLD
        }
        
        # Store results
        context['task_instance'].xcom_push(key='drift_results', value=drift_results)
        
        logger.info(f"Drift detection completed: {drift_results}")
        
        # Send alert if significant drift detected
        if requires_retraining:
            logger.warning(f"Model drift detected! Performance drift: {performance_drift:.4f}")
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        raise

def prepare_training_data(**context) -> Dict[str, Any]:
    """
    Prepare and preprocess data for model retraining
    
    Returns:
        Dictionary with data preparation results
    """
    logger.info("Preparing training data...")
    
    try:
        drift_results = context['task_instance'].xcom_pull(
            task_ids='detect_model_drift',
            key='drift_results'
        )
        
        if not drift_results['requires_retraining']:
            logger.info("No retraining required, skipping data preparation")
            return {'skipped': True, 'reason': 'No retraining required'}
        
        # Simulate data preparation process
        # In real implementation:
        # 1. Load historical data (e.g., last 12 months)
        # 2. Apply feature engineering pipeline
        # 3. Handle missing values and outliers
        # 4. Create train/validation/test splits
        # 5. Save prepared datasets
        
        preparation_results = {
            'total_samples': 50000,
            'train_samples': 35000,
            'validation_samples': 7500,
            'test_samples': 7500,
            'feature_count': 25,
            'target_distribution': {'churn': 0.15, 'retain': 0.85},
            'data_quality_score': 0.96,
            'preparation_time_seconds': 120,
            'data_path': f"{DATA_PATH}/prepared_data_{context['ds']}.parquet"
        }
        
        context['task_instance'].xcom_push(
            key='preparation_results',
            value=preparation_results
        )
        
        logger.info(f"Data preparation completed: {preparation_results['total_samples']} samples")
        return preparation_results
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

def retrain_model(**context) -> Dict[str, Any]:
    """
    Retrain the customer retention model with new data
    
    Returns:
        Dictionary with retraining results
    """
    logger.info("Starting model retraining...")
    
    try:
        # Get preparation results
        preparation_results = context['task_instance'].xcom_pull(
            task_ids='prepare_training_data',
            key='preparation_results'
        )
        
        if preparation_results.get('skipped'):
            logger.info("Skipping model retraining - no new training required")
            return {'skipped': True}
        
        # Initialize MLflow experiment
        mlflow.set_experiment("customer_retention_retraining")
        
        with mlflow.start_run(run_name=f"retraining_{context['ds']}"):
            # Log training parameters
            mlflow.log_param("training_date", context['ds'])
            mlflow.log_param("training_samples", preparation_results['train_samples'])
            mlflow.log_param("feature_count", preparation_results['feature_count'])
            
            # Simulate model training process
            # In real implementation:
            # 1. Load prepared training data
            # 2. Initialize ML pipeline
            # 3. Perform hyperparameter tuning
            # 4. Train best model
            # 5. Evaluate on validation set
            
            import time
            training_start = time.time()
            
            # Simulate training time
            time.sleep(5)  # Remove in real implementation
            
            # Simulate training results
            training_results = {
                'model_type': 'XGBoost',
                'training_accuracy': 0.912,
                'validation_accuracy': 0.905,
                'test_accuracy': 0.903,
                'training_auc': 0.945,
                'validation_auc': 0.938,
                'test_auc': 0.935,
                'training_time_seconds': time.time() - training_start,
                'best_parameters': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.9
                },
                'feature_importance_top5': {
                    'engagement_score': 0.25,
                    'recency_30d': 0.18,
                    'monetary_total_30d': 0.15,
                    'customer_tenure_months': 0.12,
                    'frequency_30d': 0.10
                },
                'model_path': f"{MODEL_PATH}/retrained_model_{context['ds']}.pkl",
                'model_version': f"v2.0_{context['ds']}"
            }
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'training_accuracy': training_results['training_accuracy'],
                'validation_accuracy': training_results['validation_accuracy'],
                'test_accuracy': training_results['test_accuracy'],
                'training_auc': training_results['training_auc'],
                'validation_auc': training_results['validation_auc'],
                'test_auc': training_results['test_auc']
            })
            
            # Log model artifacts
            mlflow.log_param("model_path", training_results['model_path'])
            mlflow.log_param("model_version", training_results['model_version'])
        
        context['task_instance'].xcom_push(
            key='training_results',
            value=training_results
        )
        
        logger.info(f"Model retraining completed - Test Accuracy: {training_results['test_accuracy']:.4f}")
        return training_results
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        raise

def validate_new_model(**context) -> Dict[str, Any]:
    """
    Validate the newly trained model against business requirements
    
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating new model...")
    
    try:
        training_results = context['task_instance'].xcom_pull(
            task_ids='retrain_model',
            key='training_results'
        )
        
        if training_results.get('skipped'):
            return {'skipped': True, 'validation_passed': False}
        
        # Validation criteria
        validation_criteria = {
            'min_accuracy': PERFORMANCE_THRESHOLD,
            'min_auc': 0.85,
            'max_accuracy_drop': 0.02,  # Compared to previous model
            'min_precision': 0.80,
            'min_recall': 0.75
        }
        
        # Get current model performance for comparison
        drift_results = context['task_instance'].xcom_pull(
            task_ids='detect_model_drift',
            key='drift_results'
        )
        
        current_baseline = drift_results.get('baseline_accuracy', 0.90)
        new_accuracy = training_results['test_accuracy']
        
        # Run validation checks
        validation_results = {
            'accuracy_check': new_accuracy >= validation_criteria['min_accuracy'],
            'auc_check': training_results['test_auc'] >= validation_criteria['min_auc'],
            'performance_regression_check': (current_baseline - new_accuracy) <= validation_criteria['max_accuracy_drop'],
            'new_model_accuracy': new_accuracy,
            'baseline_accuracy': current_baseline,
            'accuracy_improvement': new_accuracy - current_baseline,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Overall validation result
        validation_passed = all([
            validation_results['accuracy_check'],
            validation_results['auc_check'],
            validation_results['performance_regression_check']
        ])
        
        validation_results['validation_passed'] = validation_passed
        validation_results['validation_criteria'] = validation_criteria
        
        # Business impact assessment
        if validation_passed:
            # Calculate estimated business impact
            accuracy_improvement = validation_results['accuracy_improvement']
            estimated_customer_saves = int(accuracy_improvement * 10000)  # Simulate business impact
            estimated_revenue_impact = estimated_customer_saves * 1000  # $1000 per customer
            
            validation_results['business_impact'] = {
                'estimated_customer_saves': estimated_customer_saves,
                'estimated_revenue_impact': estimated_revenue_impact,
                'deployment_recommendation': 'APPROVE'
            }
        else:
            validation_results['business_impact'] = {
                'deployment_recommendation': 'REJECT',
                'rejection_reasons': []
            }
            
            if not validation_results['accuracy_check']:
                validation_results['business_impact']['rejection_reasons'].append(
                    f"Accuracy {new_accuracy:.4f} below threshold {validation_criteria['min_accuracy']}"
                )
        
        context['task_instance'].xcom_push(
            key='validation_results',
            value=validation_results
        )
        
        logger.info(f"Model validation completed - Passed: {validation_passed}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise

def deploy_model(**context) -> Dict[str, Any]:
    """
    Deploy the validated model to production
    
    Returns:
        Dictionary with deployment results
    """
    logger.info("Starting model deployment...")
    
    try:
        validation_results = context['task_instance'].xcom_pull(
            task_ids='validate_new_model',
            key='validation_results'
        )
        
        if not validation_results.get('validation_passed', False):
            logger.warning("Model validation failed, skipping deployment")
            return {
                'deployed': False,
                'reason': 'Model validation failed',
                'validation_results': validation_results
            }
        
        training_results = context['task_instance'].xcom_pull(
            task_ids='retrain_model',
            key='training_results'
        )
        
        # Deployment process
        deployment_steps = []
        
        # Step 1: Backup current model
        deployment_steps.append("Backing up current production model")
        backup_path = f"{MODEL_PATH}/backup/model_backup_{context['ds']}.pkl"
        
        # Step 2: Copy new model to production path
        deployment_steps.append("Copying new model to production")
        production_path = f"{MODEL_PATH}/customer_retention_model.pkl"
        
        # Step 3: Update model metadata
        deployment_steps.append("Updating model metadata")
        metadata = {
            'model_version': training_results['model_version'],
            'deployment_date': context['ds'],
            'baseline_accuracy': training_results['test_accuracy'],
            'model_type': training_results['model_type'],
            'training_samples': context['task_instance'].xcom_pull(
                task_ids='prepare_training_data',
                key='preparation_results'
            )['total_samples']
        }
        
        # Step 4: Restart API service (in real deployment)
        deployment_steps.append("Restarting prediction API service")
        
        # Step 5: Run smoke tests
        deployment_steps.append("Running post-deployment smoke tests")
        
        deployment_results = {
            'deployed': True,
            'deployment_date': context['ds'],
            'model_version': training_results['model_version'],
            'production_path': production_path,
            'backup_path': backup_path,
            'deployment_steps': deployment_steps,
            'new_model_accuracy': validation_results['new_model_accuracy'],
            'accuracy_improvement': validation_results['accuracy_improvement'],
            'estimated_business_impact': validation_results['business_impact']
        }
        
        context['task_instance'].xcom_push(
            key='deployment_results',
            value=deployment_results
        )
        
        logger.info(f"Model deployment completed successfully - Version: {training_results['model_version']}")
        return deployment_results
        
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise

# Create DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Automated Customer Retention Model Retraining Pipeline',
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=MAX_ACTIVE_RUNS,
    catchup=False,
    tags=['ml', 'customer-retention', 'retraining']
)

# Define tasks
extract_data_task = PythonOperator(
    task_id='extract_recent_data',
    python_callable=extract_recent_data,
    dag=dag
)

drift_detection_task = PythonOperator(
    task_id='detect_model_drift',
    python_callable=detect_model_drift,
    dag=dag
)

prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag
)

retrain_model_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag
)

validate_model_task = PythonOperator(
    task_id='validate_new_model',
    python_callable=validate_new_model,
    dag=dag
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Notification tasks
success_notification = SlackWebhookOperator(
    task_id='send_success_notification',
    http_conn_id='slack_webhook',
    message="""
✅ Customer Retention Model Retraining Completed Successfully!

🎯 New Model Performance:
- Accuracy: {{ ti.xcom_pull(task_ids='validate_new_model', key='validation_results')['new_model_accuracy'] }}
- Improvement: {{ ti.xcom_pull(task_ids='validate_new_model', key='validation_results')['accuracy_improvement'] }}

📈 Business Impact:
- Estimated Customer Saves: {{ ti.xcom_pull(task_ids='validate_new_model', key='validation_results')['business_impact']['estimated_customer_saves'] }}
- Revenue Impact: ${{ ti.xcom_pull(task_ids='validate_new_model', key='validation_results')['business_impact']['estimated_revenue_impact'] }}

🚀 Deployment Status: COMPLETED
📅 Training Date: {{ ds }}
""",
    dag=dag,
    trigger_rule='all_success'
)

failure_notification = EmailOperator(
    task_id='send_failure_notification',
    to=['data-team@company.com', 'ml-ops@company.com'],
    subject='🚨 Customer Retention Model Retraining Failed',
    html_content="""
<h2>Model Retraining Pipeline Failed</h2>

<p><strong>DAG:</strong> {{ dag.dag_id }}</p>
<p><strong>Execution Date:</strong> {{ ds }}</p>
<p><strong>Task Instance:</strong> {{ ti.task_id }}</p>

<p>Please check the Airflow logs for detailed error information.</p>

<p>Immediate action required to maintain model performance.</p>
""",
    dag=dag,
    trigger_rule='one_failed'
)

# Define task dependencies
extract_data_task >> drift_detection_task >> prepare_data_task >> retrain_model_task >> validate_model_task >> deploy_model_task

# Add notification tasks
deploy_model_task >> success_notification
[extract_data_task, drift_detection_task, prepare_data_task, retrain_model_task, validate_model_task, deploy_model_task] >> failure_notification
