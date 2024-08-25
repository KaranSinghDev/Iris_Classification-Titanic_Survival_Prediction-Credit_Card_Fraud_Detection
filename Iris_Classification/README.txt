Project Overview
This project involves developing a machine learning model to classify iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements. It includes a Flask-based web application where users can input these measurements and receive real-time predictions.

Features
Data Preprocessing: Encodes species labels and prepares the dataset for training.
Random Forest Model: Uses a Random Forest classifier to differentiate between iris species.
Flask Web Application: Allows users to input flower measurements and obtain predictions instantly.
Files
model.py: Script to train and save the Random Forest classifier.
app.py: Flask application for user interaction and prediction.
templates/index.html: HTML template for the web application interface.