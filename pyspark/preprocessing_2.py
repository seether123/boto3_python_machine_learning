from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofweek, hour, when, percentile_approx, log1p, mean, stddev
from pyspark.sql.types import IntegerType, FloatType, TimestampType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, OneHotEncoder
from pyspark.ml.stat import Correlation

# Initialize Spark Session
spark = SparkSession.builder.appName("NYC_Traffic_Preprocessing").getOrCreate()

# Load Data (No Header from S3)
file_path = "s3://your-bucket/nyc_traffic_data.csv"
df = spark.read.option("header", "false").csv(file_path)

# Assign Column Names (Since there is no header)
df = df.toDF("timestamp", "traffic_volume", "speed")

# Convert Data Types
df = df.withColumn("traffic_volume", col("traffic_volume").cast(IntegerType()))
df = df.withColumn("speed", col("speed").cast(FloatType()))
df = df.withColumn("timestamp", col("timestamp").cast(TimestampType()))

# Drop Duplicate Rows
df = df.dropDuplicates()

# Compute Median for Missing Values
median_traffic = df.approxQuantile("traffic_volume", [0.5], 0.01)[0]
median_speed = df.approxQuantile("speed", [0.5], 0.01)[0]

# Fill Missing Values with Median
df = df.fillna({"traffic_volume": median_traffic, "speed": median_speed})

# Extract Date & Time Features
df = df.withColumn("year", year(col("timestamp")))
df = df.withColumn("month", month(col("timestamp")))
df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
df = df.withColumn("hour", hour(col("timestamp")))

# Add Rush Hour Feature (1 for 7-10 AM & 4-7 PM, else 0)
df = df.withColumn("rush_hour", when((col("hour").between(7, 10)) | (col("hour").between(16, 19)), 1).otherwise(0))

# Remove Outliers using 1st & 99th Percentile
low, high = df.approxQuantile("traffic_volume", [0.01, 0.99], 0.01)
df = df.filter((col("traffic_volume") >= low) & (col("traffic_volume") <= high))

# Apply Log Transformation to Reduce Skewness
df = df.withColumn("log_traffic_volume", log1p(col("traffic_volume")))
df = df.withColumn("log_speed", log1p(col("speed")))

# Feature Interaction: Multiply Speed & Rush Hour
df = df.withColumn("speed_rush_hour_interaction", col("speed") * col("rush_hour"))

# One-Hot Encoding for Categorical Variable (Day of the Week)
encoder = OneHotEncoder(inputCols=["day_of_week"], outputCols=["day_of_week_encoded"])
df = encoder.fit(df).transform(df)

# Z-Score Normalization (Standardization)
stats = df.select(mean(col("traffic_volume")).alias("mean"), stddev(col("traffic_volume")).alias("stddev")).collect()
mean_value, stddev_value = stats[0]["mean"], stats[0]["stddev"]
df = df.withColumn("zscore_traffic", (col("traffic_volume") - mean_value) / stddev_value)

# Assemble Features for Scaling
feature_cols = ["log_traffic_volume", "log_speed", "hour", "rush_hour", "speed_rush_hour_interaction"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)

# Scale Features using MinMaxScaler
scaler = MinMaxScaler(inputCol="features_raw", outputCol="features_scaled")
df = scaler.fit(df).transform(df)

# Correlation Check: Drop Highly Correlated Features
corr_matrix = Correlation.corr(df.select("features_raw"), "features_raw").head()[0].toArray()
correlation_threshold = 0.9  # Remove features with correlation > 0.9
drop_features = []
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        if abs(corr_matrix[i][j]) > correlation_threshold:
            drop_features.append(feature_cols[j])
df = df.drop(*drop_features)

# Feature Selection: Keep Top Features
selected_features = ["features_scaled", "traffic_volume"]
df = df.select(*selected_features)

# Save Processed Data to S3 (No Header)
output_path = "s3://your-bucket/processed_nyc_traffic/"
df.write.mode("overwrite").option("header", "false").parquet(output_path)

print("✅ Data Preprocessing Completed & Saved to S3")
