import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates the model and prints metrics.
    """
    y_pred = model.predict(X_test)
    
    print(f"\n--- Evaluation for {model_name} ---")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy Score : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
    
    # Generate Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Phishing'], yticklabels=['Safe', 'Phishing'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'visualizations/confusion_matrix_{model_name}.png')
    plt.close()

def plot_dataset_distribution(df):
    plt.figure(figsize=(6,4))
    ax = sns.countplot(x='label', data=df, hue='label', palette='Set2', legend=False)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Safe', 'Phishing'])
    plt.title('Dataset Class Distribution')
    plt.tight_layout()
    plt.savefig('visualizations/dataset_distribution.png')
    plt.close()

def plot_accuracy_comparison(performances):
    plt.figure(figsize=(8,5))
    names = list(performances.keys())
    scores = list(performances.values())
    sns.barplot(x=scores, y=names, hue=names, palette='viridis', legend=False)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig('visualizations/accuracy_comparison.png')
    plt.close()

def plot_feature_importance(model, vectorizer, top_n=20):
    # Only applicable for tree-based models or models with coef_
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    
    if importances is not None:
        # Get feature names from vectorizer + metadata features
        tfidf_features = vectorizer.get_feature_names_out()
        meta_features = ['num_urls', 'has_ip', 'max_url_length', 'num_https', 'num_suspicious_domains', 'num_redirects', 'email_length', 'num_suspicious_keywords']
        all_features = np.concatenate([tfidf_features, meta_features])
        
        # In case dimensions don't match (due to some sparse matrix operations)
        if len(importances) == len(all_features):
            indices = np.argsort(importances)[::-1][:top_n]
            top_features = all_features[indices]
            top_importances = importances[indices]
            
            plt.figure(figsize=(10,6))
            sns.barplot(x=top_importances, y=top_features, hue=top_features, palette='rocket', legend=False)
            plt.title('Top 20 Feature Importances')
            plt.xlabel('Importance/Weight')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png')
            plt.close()
            print("Feature importance chart saved.")
