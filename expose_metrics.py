# # from prometheus_client import start_http_server, Counter
# # import time
# # import pickle
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.naive_bayes import MultinomialNB
# # # Define a counter metric
# # requests_counter = Counter('my_ml_model_requests_total', 'Total number of requests to my ML model')

# # # Function representing your machine learning model
# # def my_machine_learning_model(input_data):

# #   class TextModel:
# #     def __init__(self, max_features=5000):
# #         self.vectorizer = TfidfVectorizer(max_features=max_features)
# #         self.classifier = MultinomialNB() #the algorithm
        
# #     def train(self, X_train, y_train):
# #         self.classifier.fit(X_train, y_train)
        
# #     def predict(self, X_test):
# #         return self.classifier.predict(X_test)
    
# #     def save(self, vectorizer_path, model_path):
# #         pickle.dump(self.vectorizer, open(vectorizer_path, 'wb'))
# #         pickle.dump(self.classifier, open(model_path, 'wb'))
    
# #     def load(self, vectorizer_path, model_path):
# #         self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
# #         self.classifier = pickle.load(open(model_path, 'rb'))

# #     def __init__(self, vectorizer_path, model_path):
# #         self.text_model = TextModel()
# #         self.text_model.load(vectorizer_path, model_path)
    
# #     def analyze_sentiment(self, sentence):
# #         sentence_transformed = self.text_model.vectorizer.transform([sentence]).toarray()
# #         result = self.text_model.classifier.predict(sentence_transformed)[0]
# #         return 'Positive review' if result == 1 else 'Negative review'
    
# #     time.sleep(0.5)  # Simulating model processing time
          
# #   return "Prediction"

# # if __name__ == '__main__':
# #     # Start an HTTP server to expose Prometheus metrics
# #     start_http_server(8000)

# #     # Example of serving requests to your model
# #     while True:
# #         # Your model serves a request
# #         result = my_machine_learning_model("input_data")

# #         # Increment the counter metric for each request
# #         requests_counter.inc()

# #         # Simulate continuous serving of requests
# #         time.sleep(1)







# import pickle
# import numpy as np
# from prometheus_client import start_http_server, Counter, Gauge
# import time

# # Load your trained machine learning model from pickle files
# with open('star_generator.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Example metric: Counter for number of predictions
# prediction_counter = Counter('model_predictions_total', 'Total number of predictions made')
# # Example metric: Gauge for prediction latency
# prediction_latency = Gauge('prediction_latency_seconds', 'Latency of predictions')

# # Function to make predictions using your model
# def make_prediction(input_data):
#     start_time = time.time()
#     prediction = model.predict(input_data)
#     end_time = time.time()
#     prediction_latency.set(end_time - start_time)
#     prediction_counter.inc()
#     return prediction

# # HTTP server to expose Prometheus metrics
# if __name__ == '__main__':
#     start_http_server(8000)  # Prometheus metrics will be exposed on port 8000

#     # Example of continuously serving predictions
#     while True:
#         # Your input data - replace this with your actual input data
#         input_data = np.random.rand(1, 10)
        
#         # Make prediction using your model
#         prediction = make_prediction(input_data)
        
#         # Simulate continuous serving of predictions
#         time.sleep(1)


# from prometheus_client import start_http_server, Gauge
# import time
# import pandas as pd
# from classifier import TextModel
# from data_cleaning_predictions import DataPreprocessor_train
# import numpy as np


# # Create Prometheus metrics
# model_metric = Gauge('model_prediction', 'Prediction from ML model')

# # Your function to get predictions from the model

# def model_predict(filepath):
#     text_model=TextModel()
#     loaded__model,loaded_vector=text_model.load("saved_model/vectorizer_star_generator.pkl", "saved_model/star_generator.pkl")

#     df = pd.read_csv(filepath)
    
#     #data preparation
#     preprocessor = DataPreprocessor_train()
#     corpus = preprocessor.preprocess(df)
#     corpus=loaded_vector.fit_transform(corpus).toarray()
#     # print("shape = ", corpus.shape)

#     y_pred=loaded__model.predict(corpus)
#     return y_pred   
   

# # Function to update metrics
# def update_metrics():
#     while True:
#         prediction = model_predict('balanced_dataset.csv')
#         # Ensure prediction is a scalar value
#         if isinstance(prediction, (list, tuple)):
#             prediction = prediction[0]  # Take the first value
#         elif isinstance(prediction, np.ndarray):
#             if prediction.size == 1:
#                 prediction = prediction.item()  # Extract single value from numpy array
#             else:
#                 # Handle arrays with size > 1
#                 # For example, you can take the mean or median
#                 prediction = np.mean(prediction)  # Adjust as per your requirement
#         model_metric.set(float(prediction))  # Convert to float and set the value
#         time.sleep(10)  # Update interval in seconds

# if __name__ == '__main__':
#     # Start HTTP server to expose metrics
#     start_http_server(8000)
#     # Update metrics in the background
#     update_metrics()





# from prometheus_client import start_http_server, Summary
# from prometheus_client import start_http_server, Gauge
# import time
# import pandas as pd
# from classifier import TextModel
# from data_cleaning_predictions import DataPreprocessor_train
# import numpy as np

# # Define a summary metric
# REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
# model_metric = Gauge('model_prediction', 'Prediction from ML model')

# # Instrument your code to track metrics
# @REQUEST_TIME.time()
# def model_predict(filepath):
#     text_model=TextModel()
#     loaded__model,loaded_vector=text_model.load("saved_model/vectorizer_star_generator.pkl", "saved_model/star_generator.pkl")

#     df = pd.read_csv(filepath)
    
#     #data preparation
#     preprocessor = DataPreprocessor_train()
#     corpus = preprocessor.preprocess(df)
#     corpus=loaded_vector.fit_transform(corpus).toarray()
#     # print("shape = ", corpus.shape)

#     y_pred=loaded__model.predict(corpus)
#     return y_pred   
# def update_metrics():
#     while True:
#         prediction = model_predict('balanced_dataset.csv')
#         # Ensure prediction is a scalar value
#         if isinstance(prediction, (list, tuple)):
#             prediction = prediction[0]  # Take the first value
#         elif isinstance(prediction, np.ndarray):
#             if prediction.size == 1:
#                 prediction = prediction.item()  # Extract single value from numpy array
#             else:
#                 # Handle arrays with size > 1
#                 # For example, you can take the mean or median
#                 prediction = np.mean(prediction)  # Adjust as per your requirement
#         model_metric.set(float(prediction))  # Convert to float and set the value
#         time.sleep(10)  # Update interval in seconds

# # Start the HTTP server to expose metrics
# if __name__ == '__main__':
#     # Start the Prometheus HTTP server on port 8000
#     start_http_server(8000)

#     # Run your ML model
#     update_metrics()



# from prometheus_client import start_http_server, Gauge

# # Create a gauge metric for accuracy
# accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the ML model')

# # Update the accuracy metric
# accuracy_gauge.set(accuracy_value)

# if __name__ == '__main__':
#     start_http_server(8000)
#     # Run your ML model and update accuracy_gauge




from prometheus_client import start_http_server, Gauge

# Create a gauge metric for accuracy
accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the ML model')

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from star_generator_cleaning import merge_dataframes
from star_generator import StarGenerator, Evaluation
from star_generator_cleaning import CleanText
import star_generator
path1= 'balanced_dataset.csv' #BR
path2= 'datasets/books1.csv'



def model_train(path1, path2):
    #merging two datasets
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df=merge_dataframes(df1,df2,'Title')

    #cleaing column text
    cleaner = CleanText()
    df['review/text'] = df['review/text'].apply(cleaner)

    #model fitting
    star_model=StarGenerator()
    corpus= df['review/text'].tolist()
    X = star_model.vectorizer.fit_transform(corpus).toarray()
    y = df['review/score']
    
    #model training
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    star_model.train(X_train, Y_train)
    star_model.save()

    #evluating model
    predictions = star_model.predict(X_test)
    evaluator = Evaluation(Y_test, predictions)
    evaluator.calculate_metrics()
    evaluator.print_metrics()

model_train(path1, path2)

class StarGenerator:
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=max_features)
        self.classifier = MultinomialNB(alpha=0.6)

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
    def partial_train(self, X_train, y_train, classes=np.arange(1, 6)):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.partial_fit(X_train_tfidf, y_train, classes=classes)
    
    def save(self, model_filename='saved_model/star_generator.pkl', vectorizer_filename='saved_model//vectorizer_star_generator.pkl'):
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.classifier, model_file)
        with open(vectorizer_filename, 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

    def load(self, model_filename='saved_model/star_generator.pkl', vectorizer_filename='saved_model//vectorizer_star_generator.pkl'):
        self.vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
        self.classifier = pickle.load(open(model_filename, 'rb'))
        return self.classifier,self.vectorizer
class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        # self.f1 = f1_score(self.y_true, self.y_pred),   
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)

    def print_metrics(self):
        # print("F1 Score:", self.f1)
        print("Confusion Matrix:\n", self.confusion_matrix)
        print("Accuracy:", self.accuracy)
# Create an instance of StarGenerator and Evaluation

star_generator = StarGenerator()
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df=merge_dataframes(df1,df2,'Title')

    #cleaing column text
cleaner = CleanText()
df['review/text'] = df['review/text'].apply(cleaner)

    #model fitting
star_model=StarGenerator()
corpus= df['review/text'].tolist()
X = star_model.vectorizer.fit_transform(corpus).toarray()
y = df['review/score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Train the model
star_generator.train(X_train, y_train)

# Predict on the test data
y_pred = star_generator.predict(X_test)

# Calculate and set the accuracy metric in the gauge
evaluation = Evaluation(y_test, y_pred)
evaluation.calculate_metrics()
accuracy = evaluation.accuracy
accuracy_gauge.set(accuracy)

# Start the HTTP server to expose metrics
if __name__ == '__main__':
    start_http_server(8000)
