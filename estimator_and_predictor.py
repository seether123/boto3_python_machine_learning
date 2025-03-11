import sagemaker
from sagemaker.estimator import Estimator

sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::your-account-id:role/sagemaker-execution-role"

xgb_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.5-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path="s3://your-bucket/fraud-detection/models/",
    sagemaker_session=sagemaker_session
)

xgb_estimator.fit({"train": "s3://your-bucket/fraud-detection/processed/train.csv"})

predictor = xgb_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="fraud-detection-endpoint"
)

#Test deploy model
import numpy as np

test_data = np.array([[0.5, 0.1, -2.3, ..., 1.2]])  # Example transaction
response = predictor.predict(test_data.tolist())

print("Fraud Probability:", response)
