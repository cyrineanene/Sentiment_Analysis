import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class StarGenerator:
    def __init__(self):
        # Initialize to None; will be loaded from files
        self.vectorizer = None
        self.classifier = None

    def predict(self, X_test):
        # Use the classifier to predict
        return self.classifier.predict(X_test)
    
    def load(self, model_filename='saved_model/star_generator.pkl', vectorizer_filename='saved_model/vectorizer_star_generator.pkl'):
        # Correctly load the vectorizer and classifier from their respective files
        with open(vectorizer_filename, 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
        with open(model_filename, 'rb') as model_file:
            self.classifier = pickle.load(model_file)

# Function to process and predict new input text
def predict_new_text(classifier, vectorizer, text):
    # Transform the input text to the same format as the model was trained on
    X_test = vectorizer.transform([text])  # Ensure this uses transform, not fit_transform
    # Predict and return the result
    return classifier.predict(X_test)

# Main function to handle user input and model testing
def main():
    # Load the pre-trained model and vectorizer
    star_gen = StarGenerator()
    star_gen.load()  # Make sure the correct paths are used

    # Get user input
    user_input = input("Enter text for prediction: ")
    
    # Process and predict
    prediction = predict_new_text(star_gen.classifier, star_gen.vectorizer, user_input)
    
    # Output the prediction
    print(f"Predicted Class: {prediction[0]}")

if __name__ == "__main__":
    main()
