import pandas as pd
from sklearn.model_selection import train_test_split
from star_generator_cleaning import merge_dataframes
from StarGenerator.star_generator import StarGenerator, Evaluation
from star_generator_cleaning import CleanText

path1= 'datasets/balanced_dataset.csv' #BR
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