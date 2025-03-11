from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import MinMaxScaler, VectorAssembler

# Initialize Spark session
spark = SparkSession.builder.appName("FraudPreprocessing").getOrCreate()

# Load dataset from S3
s3_input_path = "s3://your-bucket/fraud-detection/raw_data.csv"
df = spark.read.csv(s3_input_path, header=True, inferSchema=True)

# Convert 'Class' (Target Label) to Integer
df = df.withColumn("Class", col("Class").cast("integer"))

# Feature Scaling - Normalize 'Amount'
vector_assembler = VectorAssembler(inputCols=["Amount"], outputCol="AmountVec")
scaler = MinMaxScaler(inputCol="AmountVec", outputCol="NormalizedAmount")

df = vector_assembler.transform(df)
df = scaler.fit(df).transform(df)

# Select only relevant columns
df = df.select(["NormalizedAmount"] + [f"V{i}" for i in range(1, 29)] + ["Class"])

# Save processed data to S3
s3_output_path = "s3://your-bucket/fraud-detection/processed/"
df.write.csv(s3_output_path, header=True, mode="overwrite")

# Stop Spark session
spark.stop()
