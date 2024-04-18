FROM python:3.10.12-slim-buster

WORKDIR /app

COPY requirement.txt  /app/requirement.txt
COPY datasets/ /app/datasets/
COPY classifier.py /app/classifier.py
COPY data_cleaning.py /app/data_cleaning.py
COPY data_extraction.py /app/data_extraction.py
COPY evaluation.py /app/evaluation.py
COPY saved_model/ /app/saved_model/
COPY model_training.py /app/model_training.py

RUN pip install -r requirement.txt  

RUN python -m nltk.downloader stopwords

CMD ["python", "model_training.py"]  