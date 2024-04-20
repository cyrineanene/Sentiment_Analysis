from star_generator_cleaning import CleanText_predict
from star_generator import StarGenerator


# Function to process and predict new input text
def predict_new_text(classifier, vectorizer, text):
    # Transform the input text to the same format as the model was trained on
    X_test = vectorizer.transform([text])  # Ensure this uses transform, not fit_transform
    # Predict and return the result
    return classifier.predict(X_test)


def main():
   
    star_gen = StarGenerator()
    star_gen.load()  

    #data preparation
    user_input = input("Enter text for prediction: ")
    clean_text=CleanText_predict(user_input)
    cleaned=clean_text.clean_text()

    # Process and predict
    prediction = predict_new_text(star_gen.classifier, star_gen.vectorizer, cleaned)
    
    # Output the prediction
    print(f"Predicted Class: {prediction[0]}")

if __name__ == "__main__":
    main()
