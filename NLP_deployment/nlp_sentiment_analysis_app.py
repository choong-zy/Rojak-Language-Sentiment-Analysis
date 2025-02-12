# streamlit run "C:\Users\choong zhi yang\OneDrive - student.tarc.edu.my\Desktop\degree RDS\y2s1\NLP\nlp_sentiment_analysis_app.py"
# run this in cmd . paste your own python file path 

import pandas as pd
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import joblib
import emoji
from mtranslate import translate
import contractions
import string
from nltk.corpus import wordnet
from spellchecker import SpellChecker #pip install pyspellchecker


# Load your pre-trained NLP model
loaded_model = joblib.load('C:/Users/choong zhi yang/OneDrive - student.tarc.edu.my/Desktop/degree RDS/y2s1/NLP/nlp_LRM_trained_model.pkl') # paste the loaded model path --> change \ to / in the path
loaded_vectorizer = joblib.load('C:/Users/choong zhi yang/OneDrive - student.tarc.edu.my/Desktop/degree RDS/y2s1/NLP/tfidf_vectorizer.pkl') # paste the loaded vectorizer path 
bm_dict = pd.read_csv('C:/Users/choong zhi yang/OneDrive - student.tarc.edu.my/Desktop/degree RDS/y2s1/NLP/malay_abbreviation.csv', encoding='latin1')  # paste the bm dictionary path


# Function to normalize Malay text using the loaded rules
def normalize_malay_text(text):
    normalization_dict = dict(zip(bm_dict['rojak'], bm_dict['actual']))# Convert the CSV data into a dictionary for easy lookup
    words = text.split()
    normalized_words = [str(normalization_dict.get(word, word)) for word in words]
    return ' '.join(normalized_words)

def is_meaningful(word): # check for meaningful word
    nltk.download('wordnet')
    synsets = wordnet.synsets(word)
    return bool(synsets)

def filter_meaningful_words(text): # return meaningful text
    words = text.split()
    meaningful_words = [word for word in words if is_meaningful(word)] # if is meaningful
    return ' '.join(meaningful_words) # join word that is meaningful

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Handle emojis
    text = emoji.demojize(text)

    # Malay Word Normalize (handle malay word shortform)
    text = normalize_malay_text(text)

    # Translation from malay to english
    text = translate(text, 'en', 'ms')

    # English Word Normalize
    text = contractions.fix(text)

    # data cleaning
    text = re.sub(r"[^A-Za-z0-9]", " ", text)  # remove character that is not an uppercase letter, lowercase letter, or digit 
    text = re.sub(r"\'s", " ", text) 
    text = re.sub(r"http\S+", " link ", text)  # Replaces URLs with the word "link".
    text = text.replace('0', 'o')
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    text = re.sub(r"\b\w\b", "", text) # Remove single characters (not part of a word)
    spell = SpellChecker()
    text = ' '.join([spell.correction(word) if spell.correction(word) is not None else word for word in text.split()]) # identify and correct the misspelled words.

    #remove meaninglesss word
    words = text.split()
    meaningful_words = [word for word in words if is_meaningful(word)] # if is meaningful
    text = ' '.join(meaningful_words) # join word that is meaningful

    # Remove punctuation using str.translate and str.maketrans
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    #remove stop word
    nltk.download('stopwords')# Download NLTK stopwords data
    stop_words = set(stopwords.words("english"))
    stop_words -= {'no', 'not', 'never'}
    tokens = word_tokenize(text)# Tokenize the text
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]# Remove stopwords
    text = ' '.join(filtered_tokens)# Join tokens back into a string

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    token = word_tokenize(text)  # Tokenize the text
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in token]
    return ' '.join(lemmatized_tokens)




def predict_sentiment(text):

    # Check if the processed text is empty or contains only stop words
    if not text or text.isspace() or len(text) < 3 :
        result = 'Unable to determine sentiment for the given input.'
    else:
        # Vectorize the text and reshape
        text_vectorized = loaded_vectorizer.transform([text]).toarray()

        # Make predictions
        # Extract sentiment and probabilities
        sentiment = loaded_model.predict(text_vectorized.reshape(1, -1))[0]

        if sentiment == 0:
            result = 'negative'
        elif sentiment == 2:
            result = 'positive'
        else:
            result = 'Unable to determine sentiment for the given input.'
    
    return result
    

# Streamlit app
st.title("Sentiment Analysis App")

# Get input from user
user_input = st.text_area("Enter text:")

# Preprocess input on button click
if st.button("Analyze"):
    processed_text = preprocess_text(user_input)
    st.write("Processed Text : ", processed_text)

    # Perform sentiment analysis
    result = predict_sentiment(processed_text)

    # Display results
    st.write("Sentiment : ", result)
