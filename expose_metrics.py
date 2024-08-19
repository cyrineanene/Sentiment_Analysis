
import os
from prometheus_client import start_http_server, Gauge
import time
# Create a gauge metric for accuracy
accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the ML model')
confusion_matrix_metric = Gauge('model_confusion_matrix', 'Confusion matrix of the ML model', ['row', 'column'])  # Updated metric definition
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from StarGenerator.star_generator_cleaning import merge_dataframes
from StarGenerator.star_generator import StarGenerator, Evaluation
from StarGenerator.star_generator_cleaning import CleanText
import StarGenerator.star_generator as star_generator
path1= 'datasets/balanced_dataset.csv' #BR

df = pd.read_csv(path1)

class StarGenerator:
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=max_features)
        self.classifier = MultinomialNB(alpha=0.6)

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
    def save(self, model_filename='./saved_model/star_generator.pkl', vectorizer_filename='./saved_model/vectorizer_star_generator.pkl'):
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.classifier, model_file)
        with open(vectorizer_filename, 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

    def load(self, model_filename='./saved_model/star_generator.pkl', vectorizer_filename='./saved_model/vectorizer_star_generator.pkl'):
        self.vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
        self.classifier = pickle.load(open(model_filename, 'rb'))
        return self.classifier, self.vectorizer

class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)

    def print_metrics(self):
        print("Confusion Matrix:\n", self.confusion_matrix)
        print("Accuracy:", self.accuracy)

def model_train(dataset):
    # Create an instance of StarGenerator and Evaluationstar_generator = StarGenerator()
    evaluation = Evaluation(None, None)  # Initialize with empty values

    # Cleaning column text
cleaner = CleanText()
df['review/text'] = df['review/text'].apply(cleaner)

    # Model fitting
star_model=StarGenerator()
corpus = df['review/text'].tolist()
X = star_model.vectorizer.fit_transform(corpus).toarray()
y = df['review/score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # Train the model
star_model.train(X_train, y_train)
star_model.save()
    # Predict on the test data
y_pred = star_model.predict(X_test)
evaluation = Evaluation(y_test, y_pred)

    # Evaluate the model
evaluation.y_true = y_test
evaluation.y_pred = y_pred
evaluation.calculate_metrics()
evaluation.print_metrics()


# Calculate and set the accuracy metric in the gauge
def update_metrics():
   evaluation = Evaluation(y_test, y_pred)
   evaluation.calculate_metrics()
   accuracy = evaluation.accuracy
   accuracy_gauge.set(accuracy)
   # Update confusion matrix metric
   confusion_matrix = evaluation.confusion_matrix
   classes = np.unique(y)  # Extract class labels
    # Iterate over confusion matrix and set each cell value
   for i, row_label in enumerate(classes):
        for j, col_label in enumerate(classes):
            cell_value = confusion_matrix[i][j]
            confusion_matrix_metric.labels(row=row_label, column=col_label).set(cell_value)
   
  
# Start the HTTP server to expose metrics
if __name__ == '__main__':
    start_http_server(8000)

    while True:
        update_metrics()
        time.sleep(10)
