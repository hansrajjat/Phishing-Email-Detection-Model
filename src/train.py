import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def train_and_select_best_model(X, y):
    """
    Trains multiple models, compares them, selects the best, and saves it.
    """
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "LinearSVC": LinearSVC(max_iter=2000, random_state=42, dual=False)
    }
    
    trained_models = {}
    performances = {}
    
    print("Training models...")
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        performances[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
    # Find best model
    best_model_name = max(performances, key=performances.get)
    best_model = trained_models[best_model_name]
    best_acc = performances[best_model_name]
    
    print(f"\nBest Model Selected: {best_model_name} with Accuracy {best_acc:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'models/best_model.pkl')
    print("Best model saved to models/best_model.pkl")
    
    return best_model_name, best_model, X_test, y_test, performances

if __name__ == "__main__":
    pass
