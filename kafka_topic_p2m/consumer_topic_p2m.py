from kafka import KafkaConsumer
import json

bootstrap_servers = 'localhost:9092'
books_topic = 'p2m'

# Create a Kafka consumer
consumer = KafkaConsumer(books_topic, bootstrap_servers=bootstrap_servers, 
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Consume messages from the Kafka topic
for message in consumer:
    books_data = message.value
    print(f"Received message from {books_topic}: {books_data}")

# Close the consumer
consumer.close()