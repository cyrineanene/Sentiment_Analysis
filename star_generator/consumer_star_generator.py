#this consumer_producer: will first receive the messages from kafka producer then processes it and makes predictions then resends the data as producer
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import os
import time
import json
from star_generator.update_model import update_model

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
    week=0
    for message in consumer:
        data_batch.append(message.value)
        # Check if batch size is reached
        if len(data_batch) >= BATCH_SIZE:
            filepath=save_batch(data_batch)
            data_batch = []  # Reset batch
            week+=1
            update_model(filepath, week)
            os.unlink(filepath)
            #week will be a multiply of 5 because each 5 weeks the model will be saved
def save_batch(data_batch):
    """
    Save a batch of data to a CSV file.
    """
    df = pd.DataFrame(data_batch)
    timestamp = int(time.time())
    filepath = os.path.join(BATCH_FOLDER, f"{CSV_FILENAME}_{timestamp}.csv")
    df.to_csv(filepath, index=False)
    return filepath

# Example usage
while True: 
    consume_data()
    time.sleep(1)