#this consumer_producer: will first receive the messages from kafka producer then processes it and makes predictions then resends the data as producer
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import os
import time
import json
from model_prediction import model_predict

# Constants
INPUT_TOPIC = 'p2m'
OUTPUT_TOPIC = 'pred'
KAFKA_SERVER = 'localhost:9092'
BATCH_SIZE = 20  # Number of messages to collect before saving to CSV
BATCH_FOLDER = 'batches'
CSV_FILENAME = 'data_batch.csv'

# Ensure batch folder exists
if not os.path.exists(BATCH_FOLDER):
    os.makedirs(BATCH_FOLDER)

def consume_data():
    """
    Consume data from a Kafka topic in JSON format.  (and saves data in a list ())
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
    # df = pd.read_csv(filepath)
    # print(df.head())
    #model_predict 
    y_pred = model_predict(filepath)
    return y_pred.tolist() 


def produce_data(y_pred):
    """
    Produce data to a Kafka topic from a DataFrame.
    """
    print('hi')
    
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    for index, row in enumerate(y_pred):
        record = {'row_index': index, 'data': row.tolist()}
    producer.send(OUTPUT_TOPIC, value=record)
    print(f"Sent message to {OUTPUT_TOPIC}: {record}")
    time.sleep(1)

    # producer.send(OUTPUT_TOPIC, value=y_pred)
    # print(f"Sent message to {OUTPUT_TOPIC}: {y_pred}")
    # time.sleep(1)
    
    

# Example usage
while True: 
    pred = consume_data()
    produce_data(pred)
    time.sleep(1)