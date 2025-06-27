import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class DataPreprocessor_train:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def preprocess(self, data):
        corpus = []
        for review in data['review']:
            review = re.sub('[^a-zA-Z]', ' ', review)
            review = review.lower()
            review = review.split()
            review = [self.stemmer.stem(word) for word in review if word not in self.stop_words]
            review = ' '.join(review)
            corpus.append(review)
        return corpus