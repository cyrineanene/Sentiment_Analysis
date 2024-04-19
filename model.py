#juste for the test



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Function to load and preprocess data
def load_data():
    import pandas as pd
    
    # movie_reviews data folder must be passed as a parameter
    movie_reviews = pd.read_csv("datasets/IMDB_Dataset.csv")  

    text, y = movie_reviews['review'], movie_reviews['sentiment']
    text = np.array(text)
    # text = [doc.decode('utf-8').replace(b"<br />", " ") for doc in text] 
    
    return text, y

# Load IMDB dataset
text, y = load_data()

# Splitting dataset into training and testing sets
text_train, text_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=42)

# Creating a Multinomial Naive Bayes classifier pipeline
model = Pipeline([
    ('vect', CountVectorizer()),  # Convert a collection of text documents to a matrix of token counts
    ('clf', MultinomialNB())      # Classifier
])

# Train the model
model.fit(text_train, y_train)
print("Model trained successfully!")

# Save the model to a file
with open('sentiment_classifier.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved to 'sentiment_classifier.pkl'")

# Load the model from a file
with open('sentiment_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Example text to classify
test_texts = ["This movie is terrible. I hate it.", "What an amazing movie! I loved it, great job!"]
predictions = loaded_model.predict(test_texts)

# Output predictions
for text, label in zip(test_texts, predictions):
    print(f"Review: '{text}' is classified as {'Positive' if label == 1 else 'Negative'}")
