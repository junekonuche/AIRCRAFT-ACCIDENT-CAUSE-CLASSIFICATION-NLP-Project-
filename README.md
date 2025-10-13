# AIRCRAFT-ACCIDENT-CAUSE-CLASSIFICATION-NLP-Project-
 Project Overview

This project uses Natural Language Processing (NLP) to classify the probable causes of aircraft accidents based on the short narrative text found in official investigation reports.
It was developed as my Machine Learning 1 Capstone Project and serves as an early step toward my planned Machine Learning 2 project on Aircraft Accident Prediction using ML.

By training a text classification model, the goal is to teach a machine to read an accident description and automatically identify the most likely cause, such as Pilot Error, Mechanical Failure, Weather Conditions, or Operational Factors.

 Objectives

Apply core supervised machine-learning techniques on unstructured text data.

Preprocess aviation-safety narratives (tokenization, cleaning, vectorization).

Build and evaluate multiple text-classification models (TF-IDF + Logistic Regression → DistilBERT).

Gain experience in feature engineering, model evaluation, and explainability.

Create a foundation for a future predictive project on aviation accident risk.

 Dataset

Dataset Name: Airplane Crashes Since 1908 – Kaggle

Features:

Date, Location, Operator, Type, Summary, Fatalities, etc.
Target Label:

Cause Category — manually or rule-based labels such as:

Pilot Error

Mechanical Failure

Weather

Operational

Other

 Project Workflow

Data Exploration

Import and inspect narratives.

Check text length, missing values, frequent keywords.

Data Cleaning & Labeling

Remove punctuation, stopwords, and lower-case text.

Create 4–5 cause labels using keyword rules or manual annotation.

Feature Extraction

Convert text to numerical features using TF-IDF Vectorizer.

Model Training

Baseline → Logistic Regression / Linear SVM.

Advanced → Fine-tuned DistilBERT model.

Evaluation

Metrics: Macro F1, Precision, Recall, Confusion Matrix.

Visualization: top words per class, misclassified examples.

Explainability

SHAP or attention visualization for model interpretation.



Macro F1-Score — balances class imbalance across multiple causes.

Precision & Recall — per-class analysis.

Confusion Matrix — to visualize misclassifications.

 Future Work

Integrate structured features (weather, aircraft type) with text narratives.

Expand label categories with multi-label classification.

Deploy a simple web app for narrative cause prediction.

Extend the work toward full Aircraft Accident Prediction using ML (planned for ML 2).

 Folder Structure
 aircraft-accident-cause-nlp
├──  README.md
├──  data/
│   └── airplane_crashes.csv
├──  notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_labeling.ipynb
│   ├── 03_tfidf_logreg_model.ipynb
│   └── 04_transformer_model.ipynb
├──  results/
│   ├── confusion_matrix.png
│   └── metrics_summary.csv
└──  requirements.txt

 Why Classification?

This is a classification problem because the target variable (“Cause Category”) consists of discrete classes such as:

Pilot Error

Mechanical Failure

Weather-related Issue

Other Causes

The model’s goal is to assign each incident to one of these categories based on its text description.

 Conclusion

This project demonstrates how NLP can be used to automate the classification of aviation incident causes.
By comparing traditional and modern approaches, we gain insights into how machine learning can enhance safety analysis and help aviation authorities identify patterns behind incidents faster and more accurately.
