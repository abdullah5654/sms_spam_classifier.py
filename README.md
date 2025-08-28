# sms_spam_classifier.py
📖 Overview

This project demonstrates how to build a complete Natural Language Processing (NLP) pipeline for text classification. The program takes raw text data, cleans and preprocesses it, converts it into numerical features using TF-IDF, and trains a machine learning model to classify text.

The example implementation uses the SMS Spam Collection Dataset to build a spam detection system.

🎯 Objectives

Clean and preprocess text data

Convert text into numerical features

Train a classification model

Evaluate the model using standard metrics

Interpret model results with word importance

⚙️ Features

Data Loading

Loads the SMS Spam dataset

Splits into training and test sets

Text Preprocessing

Lowercasing

Removing stopwords, punctuation, numbers, and special characters

Tokenization

Lemmatization

Feature Engineering

Uses TF-IDF Vectorization with a maximum of 5000 features

Model Training

Logistic Regression classifier

Hyperparameter tuning with GridSearchCV

Evaluation

Accuracy, Precision, Recall, and F1-score

Confusion Matrix visualization

Word Importance

Displays the most predictive words for spam classification

🛠️ Requirements

Python 3.x

Libraries: pandas, numpy, scikit-learn, nltk, matplotlib

You will also need the SMS Spam Collection Dataset (spam.csv).
