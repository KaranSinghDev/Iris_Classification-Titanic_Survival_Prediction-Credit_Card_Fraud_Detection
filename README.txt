# Credit Card Fraud Detection

This project aims to build a machine learning model to identify fraudulent credit card transactions. It involves preprocessing and normalizing transaction data, handling class imbalance, and training a classification algorithm to classify transactions as either fraudulent or genuine.

## Project Overview

The goal is to analyze historical credit card transaction data and develop a model that accurately detects fraudulent transactions. This project covers data preprocessing, feature scaling, handling class imbalance, model training, and evaluation.

## Dataset

- **Filename:** `creditcard.csv`
- **Features:**
  - `Time`: Time (in seconds) since the first transaction in the dataset.
  - `V1` to `V28`: PCA-transformed features (28 columns) to protect user identities.
  - `Amount`: Transaction amount.
  - `Class`: Target variable indicating fraud (1) or genuine (0).

## Project Structure

- `model.py`: Script to load, preprocess, train the model, and save it.
- `app.py`: Flask application for interacting with the trained model.
- `templates/index.html`: HTML form for user input and prediction.
- `requirements.txt`: List of required Python packages.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/CreditCardFraudDetection.git
   cd CreditCardFraudDetection
