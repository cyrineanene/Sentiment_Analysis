from kafka import KafkaProducer
import csv
import time
import json

#function to read csv file
def read_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

#book Schema 
books_schema = {
            "Id":int,
            "Title": str,
            "Price": str,
            "User_id": str,
            "profileName": str,
            "review/helpfulness": str,
            "review/score": float,
            "review/time": str,
            "review/summary": str,
            "review/text": str,     
}

if __name__ == "__main__":
    bootstrap_servers = 'localhost:9092'
    # For books data
    #topic name 
    books_topic = "books"
    #read csv file using read_csv function
    books_data = read_csv("./Model/Books_rating.csv")
    # Create a Kafka producer
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers, 
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    # Send data to Kafka topic
    for line in books_data:
        # Cast the book values based on the books_schema
        books_data_dict = {}
        for key, value in books_schema.items():
             raw_value = line[key]
             if raw_value != '' :
                books_data_dict[key] = value(raw_value)
             else:
                books_data_dict[key] = None
        # Serialize the casted record and send it to Kafka
        producer.send(books_topic, value=books_data_dict)
        print(f"Sent message to {books_topic}: {books_data_dict}")
        time.sleep(1)

    # Flush and close the producer
    producer.flush()
    producer.close()