import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

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

    def analyze_sentiment(self, sentence):
        sentence_transformed = self.vectorizer.transform([sentence]).toarray()
        result = self.classifier.predict(sentence_transformed)[0]
        return 'Positive review' if result == 1 else 'Negative review'

class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        self.f1 = f1_score(self.y_true, self.y_pred, average='binary')  
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)

    def print_metrics(self):
        print("F1 Score:", self.f1)
        print("Confusion Matrix:\n", self.confusion_matrix)
        print("Accuracy:", self.accuracy)



# class SentimentAnalyzer:
#     def __init__(self, vectorizer_path, model_path):
#         self.text_model = TextModel()
#         self.text_model.load(vectorizer_path, model_path)
    
#     def analyze_sentiment(self, sentence):
#         sentence_transformed = self.text_model.vectorizer.transform([sentence]).toarray()
#         result = self.text_model.classifier.predict(sentence_transformed)[0]
#         return 'Positive review' if result == 1 else 'Negative review'