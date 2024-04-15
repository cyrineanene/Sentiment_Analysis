import re
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def preprocess(self, data):
        data['sentiment'] = self.label_encoder.fit_transform(data['sentiment'])
        corpus = []
        for review in data['review']:
            review = re.sub('[^a-zA-Z]', ' ', review)
            review = review.lower()
            review = review.split()
            review = [self.stemmer.stem(word) for word in review if word not in self.stop_words]
            review = ' '.join(review)
            corpus.append(review)
        return corpus