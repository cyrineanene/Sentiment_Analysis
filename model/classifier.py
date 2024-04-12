import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

class SentimentAnalyzer:
    def __init__(self, vectorizer_path, model_path):
        self.text_model = TextModel()
        self.text_model.load(vectorizer_path, model_path)
    
    def analyze_sentiment(self, sentence):
        sentence_transformed = self.text_model.vectorizer.transform([sentence]).toarray()
        result = self.text_model.classifier.predict(sentence_transformed)[0]
        return 'Positive review' if result == 1 else 'Negative review'