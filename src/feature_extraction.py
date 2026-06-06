import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib

# Suspicious keywords commonly found in phishing emails
SUSPICIOUS_KEYWORDS = [
    'verify', 'password', 'login', 'urgent', 'account', 'security', 
    'click here', 'update now', 'winner', 'prize', 'overdue', 'limited',
    'bank', 'paypal', 'credit card'
]

def extract_url_features(text):
    """
    Extracts URL-based features from the raw text.
    """
    if not isinstance(text, str):
        text = ""
        
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
    num_urls = len(urls)
    has_ip = 0
    max_url_length = 0
    num_https = 0
    num_suspicious_domains = 0
    num_redirects = 0 # approximated by looking for multiple 'http' in the url
    
    for url in urls:
        # Check for IP address in URL
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            has_ip = 1
            
        max_url_length = max(max_url_length, len(url))
        
        if url.startswith('https'):
            num_https += 1
            
        # Suspicious domains or TLDs (simple heuristic)
        if '.tk' in url or '.ml' in url or 'secure' in url.lower() or 'update' in url.lower():
            num_suspicious_domains += 1
            
        # Approximation for redirects (if URL contains another http/https scheme as a parameter)
        if url.count('http') > 1:
            num_redirects += 1

    return {
        'num_urls': num_urls,
        'has_ip': has_ip,
        'max_url_length': max_url_length,
        'num_https': num_https,
        'num_suspicious_domains': num_suspicious_domains,
        'num_redirects': num_redirects
    }

def extract_text_features(text):
    """
    Extracts text-based metadata features from raw text.
    """
    if not isinstance(text, str):
        text = ""
        
    email_length = len(text)
    lower_text = text.lower()
    
    num_suspicious_keywords = sum(lower_text.count(kw) for kw in SUSPICIOUS_KEYWORDS)
    
    return {
        'email_length': email_length,
        'num_suspicious_keywords': num_suspicious_keywords
    }

def build_features(df, is_train=True, vectorizer=None):
    """
    Builds the complete feature set by combining text/URL metadata and TF-IDF features.
    Saves the trained vectorizer if is_train is True.
    """
    print("Extracting metadata features...")
    
    # Extract metadata features
    url_feats = df['text'].apply(extract_url_features).apply(pd.Series)
    txt_feats = df['text'].apply(extract_text_features).apply(pd.Series)
    
    meta_features = pd.concat([url_feats, txt_feats], axis=1).values
    
    # TF-IDF
    print("Applying TF-IDF vectorization...")
    if is_train:
        vectorizer = TfidfVectorizer(max_features=3000)
        tfidf_features = vectorizer.fit_transform(df['cleaned_text'])
        # Save the vectorizer
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        print("Vectorizer saved to models/vectorizer.pkl")
    else:
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided for inference.")
        tfidf_features = vectorizer.transform(df['cleaned_text'])
        
    # Combine sparse TF-IDF with dense metadata features
    X = sp.hstack([tfidf_features, sp.csr_matrix(meta_features)])
    
    return X, vectorizer

def extract_features_for_single_email(email_text, vectorizer, preprocessor_func):
    """
    Pipeline to extract features for a single raw email string (for inference/GUI).
    """
    # Create single-row dataframe for metadata extraction
    df_single = pd.DataFrame([{'text': email_text}])
    df_single['cleaned_text'] = df_single['text'].apply(preprocessor_func)
    
    url_feats = df_single['text'].apply(extract_url_features).apply(pd.Series)
    txt_feats = df_single['text'].apply(extract_text_features).apply(pd.Series)
    meta_features = pd.concat([url_feats, txt_feats], axis=1).values
    
    tfidf_features = vectorizer.transform(df_single['cleaned_text'])
    
    X = sp.hstack([tfidf_features, sp.csr_matrix(meta_features)])
    return X
