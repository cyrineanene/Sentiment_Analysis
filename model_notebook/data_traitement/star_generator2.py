import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def extract_values(df, columns, pattern):
 for column in columns:
        df[column] = df[column].str.extract(pattern)
 return df

# columns_to_extract = ['authors', 'categories']
# pattern = r'\'(.*)\''
# df = extract_values(df, columns_to_extract, pattern)




# Function to clean the text
def clean_text(text):
    cleaned_text = " ".join(word.lower() for word in text.split())
    return cleaned_text

# Function to remove non-word characters
def remove_non_word_chars(text):
    text_without_non_word_chars = text.str.replace('[^\w\s]', '')
    return text_without_non_word_chars

# Function to remove stopwords
def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    text_without_stopwords = " ".join(word for word in text.split() if word not in stop)
    return text_without_stopwords

# Function to remove digits
def remove_digits(text):
    text_without_digits = text.str.replace('\d+', '')
    return text_without_digits

# Function to lemmatize the text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text.split()]
    return lemmatized_text


# # Clean the text
# df['Cleaned'] = df['review/text'].apply(clean_text)

# # Remove non-word characters
# df['Cleaned'] = remove_non_word_chars(df['Cleaned'])

# # Remove stopwords
# df['Cleaned'] = remove_stopwords(df['Cleaned'])

# # Remove digits
# df['Cleaned'] = remove_digits(df['Cleaned'])

# # Lemmatize the text
# df['Cleaned'] = lemmatize_text(df['Cleaned'])