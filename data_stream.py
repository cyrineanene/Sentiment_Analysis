import tensorflow_io as tfio
from data_pipeline import consumer, producerr
from model.data_cleaning import DataPreprocessor
from model.classifier import TextModel, SentimentAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a dataset from a Kafka topic
dataset = tfio.experimental.streaming.KafkaGroupIODataset(
    topics=["p2m"],
    servers="localhost:9092",
    stream_timeout=1000  # 1 second
)

df = pd.read_csv("datasets/BR.csv")

#data preparation
preprocessor = DataPreprocessor()
corpus = preprocessor.preprocess(df)

#model NLP fitting
text_model = TextModel()
X = text_model.vectorizer.fit_transform(corpus).toarray()
y = df['review/text']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)
text_model.train(X_train, Y_train)

#model saving
text_model.save("model/saved_model/count-Vectorizer.pkl", "model/saved_model/Movies_Review_Classification.pkl")
sentiment_analyzer = SentimentAnalyzer("model/saved_model/count-Vectorizer.pkl", "model/saved_model/Movies_Review_Classification.pkl")


for message in consumer:
    # Assume model.predict returns a list of predictions
    predictions = text_model.predict(X_test)
    for result in predictions:
        producerr.send('output-topic', value=str(result))
        producerr.flush()

