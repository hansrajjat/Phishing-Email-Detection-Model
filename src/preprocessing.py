import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_data(filepath):
    """
    Loads dataset from CSV, handles missing values, and returns a DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {len(df)} records.")
        # Drop rows with missing text or labels
        df.dropna(subset=['text', 'label'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

def clean_text(text):
    """
    Performs text preprocessing:
    1. Lowercase conversion
    2. URL removal (we handle URLs in feature extraction, but clean text for NLP)
    3. Removal of special characters and punctuation
    4. Tokenization
    5. Removal of stop words
    6. Text normalization (Lemmatization)
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs (we extract them separately as features)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # 3. Remove punctuation and special characters (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. Tokenization (split by whitespace)
    tokens = text.split()
    
    # 5 & 6. Remove stop words and normalize (Lemmatization)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Rejoin words
    return ' '.join(cleaned_tokens)

def preprocess_dataframe(df):
    """
    Applies the clean_text function to the text column of the dataframe.
    """
    print("Starting text preprocessing...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Convert labels: Phishing = 1, Safe = 0
    if not pd.api.types.is_numeric_dtype(df['label']):
        df['label'] = df['label'].map({'Phishing': 1, 'Safe': 0})
        # Drop rows where mapping failed (if any weird labels existed)
        df.dropna(subset=['label'], inplace=True)
        df['label'] = df['label'].astype(int)
        
    print("Preprocessing completed.")
    return df
