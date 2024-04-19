from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Function to preprocess text
def preprocess_text(text):
    stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase
    preprocessed_text = " ".join(word.lower() for word in text.split())

    # Remove punctuation
    preprocessed_text = preprocessed_text.str.replace('[^\w\s]', '')

    # Remove stopwords
    preprocessed_text = " ".join(word for word in preprocessed_text.split() if word not in stop)

    # Remove digits
    preprocessed_text = preprocessed_text.str.replace('\d+', '')

    # Lemmatize
    preprocessed_text = [lemmatizer.lemmatize(word) for word in preprocessed_text]

    return preprocessed_text

# Function to drop NaN values and select columns
def drop_nan_and_select_columns(df):
    
    df2 = df[['Id', 'review/text']]
    df3 = df2.dropna()
    return df3


# # Drop NaN values and select columns
# df3 = drop_nan_and_select_columns(df)

# # Preprocess the text
# df3['Cleaned'] = preprocess_text(df3['review/text'])


def create_and_transform_tfidf_vectorizer(x, max_features=5000):
    
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), max_features=max_features)
    x_tfidf = tfidf_vectorizer.fit_transform(x)
    return tfidf_vectorizer, x_tfidf

# tfidf_vect_ngram, xtest_tfidf_ngram = create_and_transform_tfidf_vectorizer(x1)


# Function to make predictions using the model and save results to a CSV file
def make_predictions_and_save(model, x_tfidf, original_text_series, output_file):
    
    # Make predictions
    test_pred = model.predict(x_tfidf)
    
    # Create a DataFrame with original text and predicted scores
    df_results = pd.DataFrame({'review/text': original_text_series, 'review/score': test_pred})
    
    # Save results to a CSV file
    df_results.to_csv(output_file, index=False)


# make_predictions_and_save(model, xtest_tfidf_ngram, x1, "predictions.csv")
