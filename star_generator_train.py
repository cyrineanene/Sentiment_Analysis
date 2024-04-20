from data_extraction import extract_lines
from data_cleaning_training import DataPreprocessor
from classifier import TextModel
import pandas as pd
from sklearn.model_selection import train_test_split
from star_generator_cleaning import merge_dataframes
from star_generator import StarGenerator, Evaluation
from star_generator_cleaning import CleanText

path1= 'datasets/BR.csv'
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