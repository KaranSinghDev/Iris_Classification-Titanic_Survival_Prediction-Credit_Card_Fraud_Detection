from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Create input array for the model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict using the model
    prediction = model.predict(features)
    
    # Map the predicted value to the species name
    species = ['Setosa', 'Versicolor', 'Virginica']
    output = species[prediction[0]]
    
    return render_template('index.html', prediction_text=f'The predicted species is: {output}')

if __name__ == "__main__":
    app.run(debug=True)
