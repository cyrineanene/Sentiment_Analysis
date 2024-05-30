FROM python:3.10.12-slim-buster

EXPOSE 8080

WORKDIR /app

COPY requirement.txt  /app/requirement.txt

COPY datasets/ /app/datasets/
COPY saved_model/ /app/saved_model/

# COPY consumer_star_generator.py /app/consumer_star_generator.py
# COPY kafka_producer.py /app/kafka_producer.py

# COPY balanced_dataset.py /app/balanced_dataset.py
COPY star_generator.py /app/star_generator.py
COPY star_generator_cleaning.py /app/star_generator_cleaning.py
COPY star_generator_train.py /app/star_generator_train.py
COPY star_generator_predict.py /app/star_generator_predict.py
COPY update_model.py /app/update_model.py

RUN pip install -r requirement.txt  

RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader omw-1.4

CMD ["python3", "star_generator_predict.py"]  