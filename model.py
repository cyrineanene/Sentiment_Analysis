import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Define the data preprocessing class
class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def preprocess(self, data):
        data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        corpus = []
        for review in data['review']:
            review = re.sub('[^a-zA-Z]', ' ', review)
            review = review.lower()
            review = review.split()
            review = [self.stemmer.stem(word) for word in review if word not in self.stop_words]
            review = ' '.join(review)
            corpus.append(review)
        return corpus

# Main function to process, train, save, load, and predict
def main(csv_file_path):
    # Load data
    data = pd.read_csv(csv_file_path)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    corpus = preprocessor.preprocess(data)
    
    # Vectorization and Classifier
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(corpus, data['sentiment'], test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Save the model
    with open('sentiment_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to 'sentiment_model.pkl'")

    # Load the model
    with open('sentiment_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Predictions
    predictions = loaded_model.predict(X_test)
    prediction_texts = ["positive review" if pred == 1 else "negative review" for pred in predictions]
    
    # Print predictions
    for review, prediction in zip(X_test, prediction_texts):
        print(f"Review: '{review}' is a {prediction}")

if __name__ == "__main__":
    main("datasets/IMDB_Dataset.csv")  # specify the path to your CSV file
