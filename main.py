import os
import time

from generate_dataset import generate_mock_dataset
from src.preprocessing import load_data, preprocess_dataframe
from src.feature_extraction import build_features
from src.train import train_and_select_best_model
from src.evaluate import evaluate_model, plot_dataset_distribution, plot_accuracy_comparison, plot_feature_importance
from src.gui import launch_gui

def main():
    print("=== Phishing Email Detection System Pipeline ===")
    
    # 1. Dataset Handling
    dataset_path = 'dataset/emails.csv'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Generating mock dataset...")
        generate_mock_dataset(num_samples=1000)
        
    df = load_data(dataset_path)
    if df is None or len(df) == 0:
        print("Failed to load dataset.")
        return

    # Visualizing Dataset Distribution
    plot_dataset_distribution(df)

    # 2. Data Preprocessing
    df = preprocess_dataframe(df)

    # 3. Feature Extraction
    X, vectorizer = build_features(df, is_train=True)
    y = df['label'].values

    # 4. Model Training & Selection
    best_model_name, best_model, X_test, y_test, performances = train_and_select_best_model(X, y)

    # 5. Model Evaluation
    evaluate_model(best_model, X_test, y_test, best_model_name)
    plot_accuracy_comparison(performances)
    plot_feature_importance(best_model, vectorizer)

    print("\nPipeline execution complete. All models trained and visualizations saved.")
    
    # 6. Launch Interactive GUI
    print("\nLaunching Interactive GUI in 2 seconds...")
    time.sleep(2)
    launch_gui()

if __name__ == "__main__":
    main()
