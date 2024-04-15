# from kafka import KafkaConsumer
# import json
# from model.classifier import TextModel, SentimentAnalyzer

# # Kafka consumer setup
# bootstrap_servers = 'localhost:9092'
# books_topic = "predictions"
# consumer = KafkaConsumer(
#     books_topic, bootstrap_servers=bootstrap_servers,
#     value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# # Create an instance of the TextModel
# text_model = TextModel()
# text_model.save("model/saved_model/count-Vectorizer.pkl", "model/saved_model/Classification.pkl")
# sentiment_analyzer = SentimentAnalyzer("model/saved_model/count-Vectorizer.pkl", "model/saved_model/Classification.pkl")

# # Process messages from Kafka
# for message in consumer:
#     text = message.value['review']
#     prediction = sentiment_analyzer.analyze_sentiment(text)
#     print(f"sentiment: {prediction}")

    
# # print(f"review: {text}\nsentiment: {prediction}")
# # Close the consumer when done
# consumer.close()


from kafka import KafkaConsumer, KafkaProducer
import json
from model.classifier import SentimentAnalyzer
# Assuming SentimentAnalyzer and related classes are properly imported and configured

# Setup Kafka Consumer
consumer = KafkaConsumer(
    'predictions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Setup Kafka Producer for the results
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'))

# Load the pre-trained model
sentiment_analyzer = SentimentAnalyzer("model/saved_model/count-Vectorizer.pkl", "model/saved_model/Classification.pkl")

def analyze_and_respond(message):
    sentiment = sentiment_analyzer.analyze_sentiment(message)
    producer.send('result_topic', value={"review": message, "sentiment": sentiment})
    producer.flush()

if __name__ == "__main__":
    for message in consumer:
        review_text = message.value
        print(f"Received review for analysis: {review_text}")
        analyze_and_respond(review_text)

