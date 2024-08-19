from ClassificationModel.data_cleaning_predictions import DataPreprocessor_train
from ClassificationModel.classifier import TextModel
import pandas as pd

def model_predict(filepath):
    text_model=TextModel()
    loaded__model,loaded_vector=text_model.load("saved_model/count-Vectorizer.pkl", "saved_model/Classification.pkl")

    df = pd.read_csv(filepath)
    
    #data preparation
    preprocessor = DataPreprocessor_train()
    corpus = preprocessor.preprocess(df)
    corpus=loaded_vector.fit_transform(corpus).toarray()
    # print("shape = ", corpus.shape)

    y_pred=loaded__model.predict(corpus)
    return y_pred

#print(model_predict('/home/cyrine/Documents/SUPCOM/P2M/Sentiment_Analysis/sentiment_analysis/batches/data_batch.csv_1713526676.csv'))