import xgboost as xgb
import pandas as pd
import boto3
import argparse
import os

# Parse arguments (for SageMaker training job)
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default="/opt/ml/input/data/train.csv")
parser.add_argument("--model-dir", type=str, default="/opt/ml/model/")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.train)
X = df.drop("Class", axis=1)
y = df["Class"]

# Train XGBoost model
model = xgb.XGBClassifier(objective="binary:logistic", num_round=100)
model.fit(X, y)

# Save model to S3
model_path = os.path.join(args.model_dir, "xgboost-model.pkl")
model.save_model(model_path)
