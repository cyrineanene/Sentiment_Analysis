import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  f1_score, confusion_matrix, accuracy_score

class StarGenerator:
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=max_features)
        self.classifier = MultinomialNB(alpha=0.6)

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
    def save(self, model_filename='saved_model/star_generator.pkl', vectorizer_filename='saved_model/vectorizer_star_generator.pkl'):
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.classifier, model_file)
        with open(vectorizer_filename, 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

    def load(self, model_filename='saved_model/star_generator.pkl', vectorizer_filename='saved_model/vectorizer_star_generator.pkl'):
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