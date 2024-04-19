import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.metrics import f1_score, accuracy_score
from sklearn import model_selection, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")


# paths C:\\Users\\fathi\\Desktop\\p2m-model\\sentiment_analysis\\datasets\\BR2.csv
#       C:\\Users\\fathi\\Desktop\\p2m-model\\sentiment_analysis\\datasets\\books2.csv

def model_train(path):
    df = pd.read_csv(path)

def merge_dataframes(df1, df2, on_column):
    merged_df = pd.merge(df1, df2, on=on_column)
    return merged_df

# merged_df = merge_dataframes(df0, df6, on_column='Title')