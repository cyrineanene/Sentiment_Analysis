from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from star_generator import StarGenerator
from star_generator_cleaning import CleanText

# Function to make predictions using the model and save results to a CSV file
def make_predictions_and_save(filepath):
    star_model=StarGenerator()
    loaded__model,loaded_vector=star_model.load()

    df = pd.read_csv(filepath)

    corpus=CleanText(df)
    corpus=loaded_vector.fit_transform(corpus).toarray()
    y_pred=loaded__model.predict(corpus)
    print(y_pred)

    return y