Credit Card Fraud Detection
Project Overview
This project involves developing a machine learning model to detect fraudulent credit card transactions. It includes a Flask-based web application where users can input transaction details and receive real-time predictions on whether a transaction is fraudulent or genuine.

Features
Data Preprocessing: Normalizes transaction data and handles class imbalance using SMOTE.
Random Forest Model: Uses a Random Forest classifier to distinguish between fraudulent and genuine transactions.
Flask Web Application: Provides an interface for users to input transaction details and obtain predictions.
Files
model.py: Script to train and save the Random Forest classifier and scaler.
app.py: Flask application for user interaction and prediction.
templates/index.html: HTML template for the web application interface.