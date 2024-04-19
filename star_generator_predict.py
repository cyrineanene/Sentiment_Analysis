from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Function to split the data into training and validation sets
def split_data(x, y, test_size=0.3, random_state=2):
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return train_x, valid_x, train_y, valid_y

# Function to create a TF-IDF vectorizer
def create_tfidf_vectorizer(max_features=5000):
    
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=max_features)
    return tfidf_vectorizer

# Function to transform data using the TF-IDF vectorizer
def transform_with_tfidf_vectorizer(tfidf_vectorizer, train_x, valid_x):
    
    xtrain_tfidf = tfidf_vectorizer.fit_transform(train_x)
    xvalid_tfidf = tfidf_vectorizer.transform(valid_x)
    return xtrain_tfidf, xvalid_tfidf


# # Split the data
# train_x, valid_x, train_y, valid_y = split_data(x, y)

# # Create a TF-IDF vectorizer
# tfidf_vect_ngram = create_tfidf_vectorizer()

# # Transform the data using the TF-IDF vectorizer
# xtrain_tfidf_ngram, xvalid_tfidf_ngram = transform_with_tfidf_vectorizer(tfidf_vect_ngram, train_x, valid_x)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Function to train the Naive Bayes model
def train_naive_bayes_model(x_train, y_train, alpha=0.6):
    nb = MultinomialNB(alpha=alpha)
    model = nb.fit(x_train, y_train)
    return model

# Function to make predictions using the trained model
def make_predictions(model, x_valid):
    predictions = model.predict(x_valid)
    return predictions

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc

# Function to print accuracy score
def print_accuracy_score(acc):
    print('Accuracy of validation set is:', acc)


# # Train the Naive Bayes model
# model = train_naive_bayes_model(xtrain_tfidf_ngram, train_y)

# # Make predictions
# pred = make_predictions(model, xvalid_tfidf_ngram)

# # Calculate accuracy
# acc = calculate_accuracy(valid_y, pred)

# # Print accuracy score
# print_accuracy_score(acc)