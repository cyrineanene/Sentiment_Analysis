# from kafka import KafkaProducer
# import csv
# import time
# import json
# from model_training import model_train

# #function to read csv file
# def read_csv(file_path):
#     data = []
#     with open(file_path, newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             data.append(row)
#     return data

# #book Schema 

# books_schema = { #just for the test 
#     "review":str,
#     "sentiment":str,
# }

# if __name__ == "__main__":
#     bootstrap_servers = 'localhost:9092'
#     books_topic = "predictions"
#     #read csv file using read_csv function
#     books_data = read_csv('datasets/imbd_less.csv') 
#     # Create a Kafka producer
#     producer = KafkaProducer(bootstrap_servers=bootstrap_servers, 
#                              value_serializer=lambda v: json.dumps(v).encode('utf-8'))
#     sentiment_analyzer = model_train(books_data)
#     # Send data to Kafka topic
#     for line in books_data:
#         # Cast the book values based on the books_schema
#         books_data_dict = {}
#         for key, value in books_schema.items():
#             books_data_dict[key] = sentiment_analyzer.analyze_sentiment(line[key])
#             #  raw_value = line[key]
#             #  if raw_value != '' :
#             #     books_data_dict[key] = value(raw_value)
#             #  else:
#             #     books_data_dict[key] = None
#         # Serialize the casted record and send it to Kafka
#         producer.send(books_topic, value=books_data_dict)
#         print(f"Sent message to {books_topic}: {books_data_dict}")
#         time.sleep(1)

#     # Flush and close the producer
#     producer.flush()
#     producer.close()

from kafka import KafkaProducer
import csv
import time
import json

# Function to read CSV file and return data as a list of dictionaries
def read_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Kafka Producer configuration
bootstrap_servers = 'localhost:9092'
books_topic = "predictions"

# Create a Kafka producer for sending JSON data
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers, 
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Function to read and send data to Kafka topic
def send_data_to_kafka(file_path, topic):
    # Read data from the CSV file
    books_data = read_csv(file_path)
    
    # Send each row of data as a message to the Kafka topic
    for line in books_data:
        producer.send(topic, value=line)
        print(f"Sent message to {topic}: {line}")
        time.sleep(1)  # Sleep for 1 second to simulate data streaming

    # Ensure all messages are sent and then close the producer
    producer.flush()
    producer.close()

# Sending data from the specified CSV file to the Kafka topic
send_data_to_kafka('datasets/imbd_less.csv', books_topic)

