import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.model import Model
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TuningStep, RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics

# ===========================
# 1. AWS SageMaker Setup
# ===========================
sagemaker_session = sagemaker.Session()
role = get_execution_role()
region = boto3.Session().region_name
bucket = "your-sagemaker-bucket"
prefix = "nyc-traffic-xgboost"

# Data paths in S3
train_data_s3 = f"s3://{bucket}/{prefix}/train/"
test_data_s3 = f"s3://{bucket}/{prefix}/test/"
output_path = f"s3://{bucket}/{prefix}/models/"

# ===========================
# 2. Define XGBoost Model
# ===========================
xgboost_train = XGBoost(
    entry_point="train.py",  # XGBoost training script
    framework_version="1.5-1",
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    output_path=output_path,
    hyperparameters={"objective": "reg:squarederror", "num_round": 100}
)

# ===========================
# 3. Hyperparameter Tuning
# ===========================
hyperparameter_ranges = {
    "max_depth": IntegerParameter(3, 10),
    "learning_rate": ContinuousParameter(0.01, 0.2),
    "min_child_weight": IntegerParameter(1, 10),
    "subsample": ContinuousParameter(0.5, 1.0),
    "colsample_bytree": ContinuousParameter(0.5, 1.0)
}

tuner = HyperparameterTuner(
    estimator=xgboost_train,
    objective_metric_name="validation:rmse",
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[{"Name": "validation:rmse", "Regex": "validation-rmse:([0-9\\.]+)"}],
    max_jobs=10,
    max_parallel_jobs=2
)

train_input = TrainingInput(train_data_s3, content_type="parquet")
test_input = TrainingInput(test_data_s3, content_type="parquet")

tuning_step = TuningStep(
    name="XGBoostHyperparameterTuning",
    tuner=tuner,
    inputs={"train": train_input, "validation": test_input}
)

# ===========================
# 4. Register Best Model
# ===========================
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=f"{output_path}/best-model/metrics.json",
        content_type="application/json"
    )
)

best_model = Model(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, "1.5-1"),
    model_data=tuner.best_estimator().model_data,
    role=role
)

register_model_step = RegisterModel(
    name="RegisterXGBoostModel",
    model=best_model,
    content_types=["application/x-parquet"],
    response_types=["application/json"],
    model_metrics=model_metrics
)

# ===========================
# 5. Deploy Model as Endpoint
# ===========================
predictor = tuner.best_estimator().deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="xgboost-nyc-traffic-predictor"
)

print(f"✅ Model Deployed at Endpoint: {predictor.endpoint_name}")

# ===========================
# 6. Define and Execute Pipeline
# ===========================
pipeline = Pipeline(
    name="NYCTrafficPredictionPipeline",
    steps=[tuning_step, register_model_step]
)

pipeline.upsert(role_arn=role)
pipeline.start()
print("🚀 SageMaker Pipeline Execution Started!")
