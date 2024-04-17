from data_extraction import extract_lines
from data_cleaning import DataPreprocessor
from classifier import TextModel, SentimentAnalyzer
from evaluation import Evaluation
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/IMDB_Dataset.csv')

def model_train(df):
#data preparation
    preprocessor = DataPreprocessor()
    corpus = preprocessor.preprocess(df)

#model NLP fitting
    text_model = TextModel()
    X = text_model.vectorizer.fit_transform(corpus).toarray()
    y = df['sentiment']

#training model
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    text_model.train(X_train, Y_train)

#model evaluation
    # model_evaluation = Evaluation(Y_train, text_model.predict(X_train))
    # model_evaluation.calculate_metrics()
    # model_evaluation.print_metrics()

#model saving
    text_model.save("saved_model/count-Vectorizer.pkl", "saved_model/Classification.pkl")
    sentiment_analyzer = SentimentAnalyzer("saved_model/count-Vectorizer.pkl", "saved_model/Classification.pkl")
    return sentiment_analyzer

model= model_train(df)

#printing results 
# for i in range(3):
#     ch=input('What did you think of the move The notebook?')
#     print(sentiment_analyzer.analyze_sentiment(ch))
# reviews=list()
# for i in range(6):
#     reviews.append(corpus[i])
# for review in reviews:
#     print(sentiment_analyzer.analyze_sentiment(review))

#will be used for the star generator model later
# extract_lines('datasets/books_data.csv', 'datasets/books1.csv', 100)
# df1 = pd.read_csv("datasets/books1.csv")
# extract_lines('datasets/Books_rating.csv', 'datasets/books2.csv', 100)
# df2 = pd.read_csv("datasets/books2.csv")