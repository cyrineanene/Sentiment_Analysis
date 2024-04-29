FROM python:3.10.12-slim-buster

EXPOSE 8080

WORKDIR /app

COPY requirement.txt  /app/requirement.txt

COPY datasets/ /app/datasets/
COPY saved_model/ /app/saved_model/

# COPY classifier.py /app/classifier.py
# COPY data_cleaning_training.py /app/data_cleaning_training.py
# COPY data_cleaning_predictions.py /app/data_cleaning_predictions.py
# COPY data_extraction.py /app/data_extraction.py
# COPY model_training.py /app/model_training.py
# COPY model_prediction.py /app/model_prediction.py

# COPY consumer_producer.py /app/consumer_producer.py
COPY consumer_star_generator.py /app/consumer_star_generator.py
COPY kafka_producer.py /app/kafka_producer.py

COPY balanced_dataset.py /app/balanced_dataset.py
COPY star_generator.py /app/star_generator.py
COPY star_generator_cleaning.py /app/star_generator_cleaning.py
COPY star_generator_train.py /app/star_generator_train.py
COPY star_generator_predict.py /app/star_generator_predict.py
COPY update_model.py /app/update_model.py

RUN pip install -r requirement.txt  

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

CMD ["python", "kafka_producer.py", "consumer_star_generator.py"]  