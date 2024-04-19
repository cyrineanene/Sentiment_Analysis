from data_extraction import extract_lines
from data_cleaning_training import DataPreprocessor
from classifier import TextModel
import pandas as pd
from sklearn.model_selection import train_test_split
from star_generator_cleaning import merge_dataframes
from star_generator import StarGenerator
from star_generator_cleaning import CleanText

path1= 'datasets/BR.csv'
path2= 'datasets/books1.csv'

def model_train(path1, path2):
    star_model=StarGenerator()

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df=merge_dataframes(df1,df2,'Title')

    df['review/text'] = df['review/text'].apply(CleanText(df).clean_text)
    X =star_model.vectorizer.fit_transform(corpus).toarray()
#should be continued this is the train function
    

#     #data preparation
#     preprocessor = DataPreprocessor()
#     corpus = preprocessor.preprocess(df)

#     #model NLP fitting
#     text_model = TextModel()
#     X = text_model.vectorizer.fit_transform(corpus).toarray()
#     y = df['sentiment']

#     #training model
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#     text_model.train(X_train, Y_train)

#     #model saving
#     df['sentiment'] = df['review'].apply(text_model.analyze_sentiment) #lezm el colonne taa texte esmha ykoun texte
#     text_model.save("saved_model/count-Vectorizer.pkl", "saved_model/Classification.pkl")

# model_train(path)