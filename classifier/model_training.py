from data_extraction.data_extraction import extract_lines
from classifier.data_cleaning_training import DataPreprocessor
from classifier.classifier import TextModel
import pandas as pd
from sklearn.model_selection import train_test_split

path= 'datasets/IMDB_Dataset.csv'

def model_train(path):
    df = pd.read_csv(path)

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

    #model saving
    # df['sentiment'] = df['review'].apply(text_model.analyze_sentiment) #nahitha kater hasit zeyda ajouteha ken hasitha lezma wala khdemch
    text_model.save("saved_model/count-Vectorizer.pkl", "saved_model/Classification.pkl")

model_train(path)