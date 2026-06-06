# Phishing Email Detection System

An end-to-end Machine Learning project in Python to classify emails as "Phishing" or "Safe" using Scikit-learn.

## Project Features
- **Data Handling**: Loads datasets, handles missing values, and visualizes distributions.
- **Preprocessing**: Lowercases, removes punctuation, removes stop words, and normalizes text.
- **Feature Extraction**: Extracts URL-based features and keyword features, along with TF-IDF vectorization.
- **Model Training & Selection**: Trains `MultinomialNB`, `LogisticRegression`, `RandomForestClassifier`, and `LinearSVC`. Selects the best performing model.
- **Evaluation**: Displays accuracy, precision, recall, F1 score, classification reports, and confusion matrices.
- **GUI**: Interactive Tkinter interface to paste emails and see predictions with confidence scores.

## Installation
1. Ensure you have Python 3.10+ installed.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure your dataset (`emails.csv`) is placed inside the `dataset/` directory.

## Usage

### Training the Models
To execute the full training pipeline (loading data, extracting features, training, evaluating, and launching the GUI), run the main script:
```bash
python main.py
```
*(Note: If your dataset is large, this may take a few minutes to complete.)*

This script will:
- Load and preprocess the dataset.
- Train multiple models and select the best one (like RandomForestClassifier).
- Save the best model and vectorizer to the `models/` directory.
- Generate evaluation plots in the `visualizations/` directory.
- Launch the Tkinter application automatically.

### Quick Launch (GUI Only)
Once your models are already trained and saved in the `models/` folder, you can skip the training process and instantly launch the GUI anytime by running:
```bash
python -m src.gui
```
