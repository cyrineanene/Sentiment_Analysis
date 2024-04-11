from model.data_extraction import extract_lines
from model.data_cleaning import DataPreprocessor, TextModel, SentimentAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split

extract_lines('datasets/IMDB_Dataset.csv', 'datasets/newbooks.csv', 10000)
df = pd.read_csv("datasets/newbooks.csv")
    
preprocessor = DataPreprocessor()
corpus = preprocessor.preprocess(df)
    
text_model = TextModel()
X = text_model.vectorizer.fit_transform(corpus).toarray()
y = df['sentiment']
    
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)
text_model.train(X_train, Y_train)
    
text_model.save("model/saved_model/count-Vectorizer.pkl", "model/saved_model/Movies_Review_Classification.pkl")
    
sentiment_analyzer = SentimentAnalyzer("model/saved_model/count-Vectorizer.pkl", "model/saved_model/Movies_Review_Classification.pkl")
    
#reviews = [
#        'bad',
 #       'the best'
  #  ]
    
#for review in reviews:
 #       print(sentiment_analyzer.analyze_sentiment(review))
