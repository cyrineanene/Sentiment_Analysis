from prometheus_client import start_http_server, Counter
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# Define a counter metric
requests_counter = Counter('my_ml_model_requests_total', 'Total number of requests to my ML model')

# Function representing your machine learning model
def my_machine_learning_model(input_data):

  class TextModel:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.classifier = MultinomialNB() #the algorithm
        
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
    def save(self, vectorizer_path, model_path):
        pickle.dump(self.vectorizer, open(vectorizer_path, 'wb'))
        pickle.dump(self.classifier, open(model_path, 'wb'))
    
    def load(self, vectorizer_path, model_path):
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        self.classifier = pickle.load(open(model_path, 'rb'))

    def __init__(self, vectorizer_path, model_path):
        self.text_model = TextModel()
        self.text_model.load(vectorizer_path, model_path)
    
    def analyze_sentiment(self, sentence):
        sentence_transformed = self.text_model.vectorizer.transform([sentence]).toarray()
        result = self.text_model.classifier.predict(sentence_transformed)[0]
        return 'Positive review' if result == 1 else 'Negative review'
    
    time.sleep(0.5)  # Simulating model processing time
          
  return "Prediction"

if __name__ == '__main__':
    # Start an HTTP server to expose Prometheus metrics
    start_http_server(8000)

    # Example of serving requests to your model
    while True:
        # Your model serves a request
        result = my_machine_learning_model("input_data")

        # Increment the counter metric for each request
        requests_counter.inc()

        # Simulate continuous serving of requests
        time.sleep(1)
