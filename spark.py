from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
from model_training import model_train

# Initialize Spark Session
spark = SparkSession.builder \
    .appName('sentiment model consumer') \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Example to load data directly from Kafka (if your consumer pushes it into a topic that Spark reads)
df = spark \
    .read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "p2m") \
    .load()

# We need to convert the binary 'value' column into a string then to JSON
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType

schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("Title", StringType(), True),
    StructField("Price", StringType(), True),
    StructField("User_id", StringType(), True), 
    StructField("profileName", StringType(), True),
    StructField("review/helpfulness", StringType(), True),
    StructField("review/score", DoubleType(), True),
    StructField("review/time", StringType(), True),
    StructField("review/summary", StringType(), True),
    StructField("review/text", StringType(), True),         
])

json_df = df.select(
    from_json(col("value").cast("string"), schema).alias("parsed_value")
)

# Selecting the features column and converting to DenseVector 
from pyspark.sql.functions import udf
from classifier import analyze_sentiment

model = model_train(df)
sentiment_analysis_udf = udf(analyze_sentiment, StringType())

# Apply UDF to DataFrame
result_df = json_df.withColumn("sentiment", sentiment_analysis_udf(col("review/text")))

# Show results
result_df.select("Id", "Title", "sentiment").show(truncate=False)