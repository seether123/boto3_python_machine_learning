from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofweek, hour, when, percentile_approx, log1p
from pyspark.sql.types import IntegerType, FloatType, TimestampType
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler, OneHotEncoder, PCA

# Initialize Spark Session
spark = SparkSession.builder.appName("NYC_Traffic_Preprocessing").getOrCreate()

# Load Data (CSV from S3, No Header)
file_path = "s3://your-bucket/nyc_traffic_data.csv"
df = spark.read.option("header", "false").csv(file_path)

# Assign Column Names (No Header in Data)
df = df.toDF("timestamp", "traffic_volume", "speed", "road_type")

# Convert Data Types
df = df.withColumn("traffic_volume", col("traffic_volume").cast(IntegerType()))
df = df.withColumn("speed", col("speed").cast(FloatType()))
df = df.withColumn("timestamp", col("timestamp").cast(TimestampType()))

# Drop Duplicate Rows
df = df.dropDuplicates()

# Compute Median Values for Missing Data
median_traffic = df.approxQuantile("traffic_volume", [0.5], 0.01)[0]
median_speed = df.approxQuantile("speed", [0.5], 0.01)[0]

# Fill Nulls with Median Values
df = df.fillna({"traffic_volume": median_traffic, "speed": median_speed})

# Extract Date & Time Features
df = df.withColumn("year", year(col("timestamp")))
df = df.withColumn("month", month(col("timestamp")))
df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
df = df.withColumn("hour", hour(col("timestamp")))

# Add Rush Hour Feature (1 for 7-10 AM & 4-7 PM, else 0)
df = df.withColumn("rush_hour", when((col("hour").between(7, 10)) | (col("hour").between(16, 19)), 1).otherwise(0))

# Remove Outliers (1st & 99th Percentile)
low, high = df.approxQuantile("traffic_volume", [0.01, 0.99], 0.01)
df = df.filter((col("traffic_volume") >= low) & (col("traffic_volume") <= high))

# Handle Categorical Variables (One-Hot Encoding for 'road_type')
encoder = OneHotEncoder(inputCol="road_type", outputCol="road_type_encoded")
df = encoder.fit(df).transform(df)

# Handle Skewed Data (Log Transformation for Traffic Volume)
df = df.withColumn("traffic_volume_log", log1p(col("traffic_volume")))

# Assemble Features for Scaling
feature_cols = ["traffic_volume_log", "speed", "rush_hour", "road_type_encoded"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)

# Min-Max Scaling (For XGBoost)
minmax_scaler = MinMaxScaler(inputCol="features_raw", outputCol="features_scaled")
df = minmax_scaler.fit(df).transform(df)

# Standard Scaling (For Robustness)
standard_scaler = StandardScaler(inputCol="features_raw", outputCol="features_standardized", withMean=True, withStd=True)
df = standard_scaler.fit(df).transform(df)

# PCA for Dimensionality Reduction (Reduce to 3 Principal Components)
pca = PCA(k=3, inputCol="features_standardized", outputCol="pca_features")
df = pca.fit(df).transform(df)

# Remove Columns with Zero Variance (Constant Features)
summary = df.describe()
zero_variance_cols = [col for col in df.columns if summary.select(col).distinct().count() == 1]
df = df.drop(*zero_variance_cols)

# Select Required Columns (No Headers)
df = df.select("pca_features", "traffic_volume")

# Save to S3 (Without Header)
output_path = "s3://your-bucket/processed_nyc_traffic/"
df.write.mode("overwrite").option("header", "false").parquet(output_path)

print("✅ Data Preprocessing Completed & Saved to S3")
