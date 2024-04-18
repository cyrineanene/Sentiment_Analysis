#this consumer_producer: will first receive the messages from kafka producer then processes it and makes predictions then resends the data as producer
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import os
import time
import json

# Constants
INPUT_TOPIC = 'p2m'
OUTPUT_TOPIC = 'predictions'
KAFKA_SERVER = 'localhost:9092'
BATCH_SIZE = 100  # Number of messages to collect before saving to CSV
BATCH_FOLDER = 'batches'
CSV_FILENAME = 'data_batch.csv'

# Ensure batch folder exists
if not os.path.exists(BATCH_FOLDER):
    os.makedirs(BATCH_FOLDER)

def consume_data():
    """
    Consume data from a Kafka topic in JSON format.
    """
    # Create a Kafka consumer
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    data_batch = []
    for message in consumer:
        data_batch.append(message.value)
        # Check if batch size is reached
        if len(data_batch) >= BATCH_SIZE:
            save_batch(data_batch)
            data_batch = []  # Reset batch

def save_batch(data_batch):
    """
    Save a batch of data to a CSV file.
    """
    df = pd.DataFrame(data_batch)
    timestamp = int(time.time())
    filepath = os.path.join(BATCH_FOLDER, f"{CSV_FILENAME}_{timestamp}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved batch to {filepath}")
    process_data(filepath)

def process_data(filepath):
    """
    Load data from CSV, apply model predictions, and produce results to Kafka.
    """
    df = pd.read_csv(filepath)
    from model_training import model_train
    result_df = model_train(df)
    produce_data(result_df, 'batches/data_batch.csv')

def read_csv(file_path):
    import csv
    """
    Safely read a CSV file into a DataFrame.
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                return [row for row in reader]
        else:
            print(f"Error: The file {file_path} does not exist.")
            return []
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return []

def produce_data(books_data,csv_path):
    books_data = read_csv(csv_path)
    """
    Produce data to a Kafka topic from a DataFrame.
    """
    books_schema = {
    "Id": int, "Title": str, "Price": str, "User_id": str, "profileName": str,
    "review/helpfulness": str, "review/score": float, "review/time": str,
    "review/summary": str, "review/text": str
}
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    for line in books_data:
        books_data_dict = {}
        for key, value in books_schema.items():
            raw_value = line.get(key, '')
            books_data_dict[key] = value(raw_value) if raw_value != '' else None
        producer.send(OUTPUT_TOPIC, value=books_data_dict)
        print(f"Sent message to {OUTPUT_TOPIC}: {books_data_dict}")
        time.sleep(1)
    
    producer.flush()
    producer.close()

# Example usage
consume_data()