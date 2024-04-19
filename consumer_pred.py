from kafka import KafkaConsumer
import json

bootstrap_servers = 'localhost:9092'
books_topic = 'predictions'

# Create a Kafka consumer
consumer = KafkaConsumer(books_topic, bootstrap_servers=bootstrap_servers, 
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Consume messages from the Kafka topic
print(consumer)
for message in consumer:
    print(message)
    preds = message.value
    print("hi")
    print(f"Received message from {books_topic}: {preds}")

# Close the consumer
consumer.close()
