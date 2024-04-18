from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
from model_training import model_train

# Initialize Spark Session
spark = SparkSession.builder \
    .appName('sentiment model consumer') \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
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
    StructField("sentiment", StringType(), True),
])

        # Show data
json_df = df.select(
    from_json(col("value").cast("string"), schema).alias("books_data") 
)
# json_df.show()
# # Selecting the features column and converting to DenseVector 
from pyspark.sql.functions import udf
from classifier import TextModel
# json_df.write.csv('dataset/spark.csv', header=True, mode='overwrite')

model=TextModel()

# model = model_train('dataset/spark.csv')
sentiment_analysis_udf = udf(model.analyze_sentiment, StringType())

from pyspark.sql.functions import lit

# # Apply UDF to DataFrame
result_df = json_df.withColumn("books_data.sentiment", lit(sentiment_analysis_udf(col("books_data.review/text"))))

# Show results
result_df.select("books_data.Id", "books_data.review/text", "books_data.sentiment").show(truncate=False)