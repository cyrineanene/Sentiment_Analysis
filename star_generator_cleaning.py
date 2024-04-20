import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

class CleanText:
    def __call__(self, text):
        self.text = text
        return self.clean_text()    #Make the instance callable to directly accept text from DataFrame.apply
    
    def clean_text(self):
        # Lowercase and split the text
        cleaned_text = " ".join(word.lower() for word in self.text.split())
        # Remove non-word characters
        text_without_non_word_chars = re.sub('[^\w\s]', '', cleaned_text)
        # Get English stopwords
        stop = set(stopwords.words('english'))
        # Remove stopwords
        text_without_stopwords = " ".join(word for word in text_without_non_word_chars.split() if word not in stop)
        # Remove digits
        text_without_digits = re.sub('\d+', '', text_without_stopwords)
        # Initialize the lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Lemmatize text
        lemmatized_text = [lemmatizer.lemmatize(word) for word in text_without_digits.split()]
        # Return a string of lemmatized text
        return ' '.join(lemmatized_text)

class CleanText_predict:
    def __init__(self, text):
        self.text = text
    
    def clean_text(self):
        # Lowercase and split the text
        cleaned_text = " ".join(word.lower() for word in self.text.split())
        # Remove non-word characters
        text_without_non_word_chars = re.sub('[^\w\s]', '', cleaned_text)
        # Get English stopwords
        stop = set(stopwords.words('english'))
        # Remove stopwords
        text_without_stopwords = " ".join(word for word in text_without_non_word_chars.split() if word not in stop)
        # Remove digits
        text_without_digits = re.sub('\d+', '', text_without_stopwords)
        # Initialize the lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Lemmatize text
        lemmatized_text = [lemmatizer.lemmatize(word) for word in text_without_digits.split()]
        # Return a string of lemmatized text
        return ' '.join(lemmatized_text)


#hethy fi training hachty beha
def merge_dataframes(df1, df2, on_column):
    merged_df = pd.merge(df1, df2, on=on_column)
    return merged_df

# Example usage
# cleaner = CleanText("This is a sample Text with numbers 123 and stopwords.")
# print(cleaner.clean_text())