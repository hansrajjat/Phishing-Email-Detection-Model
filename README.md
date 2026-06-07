# Phishing Email Detection System

An end-to-end Machine Learning project in Python to classify emails as "Phishing" or "Safe" using Scikit-learn.

## 📊 Dataset & Model Performance
This project utilizes a massive real-world dataset comprising over **164,000** emails. The dataset includes a diverse mix of legitimate business communications ("Safe") and malicious phishing attempts. 

Through rigorous training and testing, the **RandomForestClassifier** emerged as the best-performing model, achieving:
- **Accuracy**: 99.44%
- **Precision**: 99.44%
- **Recall**: 99.50%
- **F1-Score**: 99.47%

## ⚙️ Feature Engineering
The system does not just look at the raw text. It extracts a combination of advanced features to make its predictions:
1. **Textual Semantic Features**: Uses `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) capped at 3000 features to understand the linguistic structure and vocabulary of the email.
2. **URL Heuristics**: Extracts all URLs from the email body and analyzes them for:
   - Presence of raw IP addresses (e.g., `http://192.168.1.1`).
   - Usage of suspicious Top Level Domains (TLDs) or keywords (`.tk`, `update`, `secure`).
   - Anomalous URL lengths and multiple redirect structures.
3. **Keyword Density**: Counts the frequency of high-risk keywords commonly found in social engineering attacks (e.g., "urgent", "password", "verify account").

## 🚀 Project Features
- **Data Handling**: Loads massive datasets seamlessly, handles missing values, and visualizes distributions using pandas and seaborn.
- **Preprocessing**: Lowercases, removes punctuation, removes stop words, and normalizes text via lemmatization (NLTK).
- **Model Training & Selection**: Automatically trains `MultinomialNB`, `LogisticRegression`, `RandomForestClassifier`, and `LinearSVC`. Dynamically selects and serializes the best performing model using `joblib`.
- **Evaluation**: Automatically generates confusion matrix heatmaps, accuracy comparison charts, and feature importance bar graphs.
- **GUI**: A modern, interactive Tkinter interface allowing you to paste any email and see real-time predictions, confidence scores, and risk levels.

## 📁 Project Structure
```text
Phishing_Email_Detection/
│
├── dataset/                   Contains the emails.csv dataset (Not tracked by git)
├── models/                    Serialized best_model.pkl and vectorizer.pkl
├── src/                       Source code modules
│   ├── preprocessing.py       Text cleaning and lemmatization
│   ├── feature_extraction.py  TF-IDF and URL heuristic extraction
│   ├── train.py               ML training and model selection pipeline
│   ├── evaluate.py            Metrics calculation and matplotlib visualizations
│   ├── predict.py             Inference logic for single emails
│   └── gui.py                 Tkinter graphical user interface
├── visualizations/            Generated charts (Confusion Matrix, Distributions)
├── generate_dataset.py        Fallback script to generate mock data if CSV is missing
├── main.py                    Orchestrator script to run the full pipeline
└── requirements.txt           Python dependencies
```

## 🛠️ Installation
1. Ensure you have Python 3.10+ installed.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure your dataset (`emails.csv`) is placed inside the `dataset/` directory.

## 💻 Usage

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

## 👨‍💻 Developed By
Hansraj Jat.

## 📜 License
This project is licensed under the [MIT License](LICENSE).
