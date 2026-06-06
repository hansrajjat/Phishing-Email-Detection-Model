import joblib
import numpy as np
from src.preprocessing import clean_text
from src.feature_extraction import extract_features_for_single_email

def load_saved_artifacts():
    """
    Loads the trained model and vectorizer.
    """
    try:
        model = joblib.load('models/best_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error loading models: {e}. Please train the model first.")
        return None, None

def predict_email(email_text):
    """
    Predicts whether a single email is PHISHING or SAFE.
    Returns the prediction, confidence score, and probability percentage.
    """
    model, vectorizer = load_saved_artifacts()
    
    if model is None or vectorizer is None:
        return "ERROR", 0.0, 0.0
        
    # Extract features using the pipeline
    X_input = extract_features_for_single_email(email_text, vectorizer, clean_text)
    
    # Predict
    prediction_num = model.predict(X_input)[0]
    
    # Probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_input)[0]
        confidence = max(probs)
        prob_phishing = probs[1]
    else:
        # LinearSVC and similar models without predict_proba by default
        decision = model.decision_function(X_input)[0]
        # Approximation to probability using sigmoid for display purposes
        prob_phishing = 1 / (1 + np.exp(-decision))
        probs = [1 - prob_phishing, prob_phishing]
        confidence = max(probs)
        
    status = "PHISHING" if prediction_num == 1 else "SAFE"
    confidence_pct = confidence * 100
    prob_phishing_pct = prob_phishing * 100
    
    return status, confidence_pct, prob_phishing_pct
