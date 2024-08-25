# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')  # Save and load scaler if needed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input from form
        time = float(request.form['Time'])
        amount = float(request.form['Amount'])
        features = [time] + [float(request.form[f'V{i}']) for i in range(1, 29)] + [amount]
        
        # Preprocess input data
        input_data = np.array(features).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Normalize if scaler was used
        
        # Predict
        prediction = model.predict(input_data)
        return render_template('index.html', prediction='Fraudulent' if prediction[0] == 1 else 'Genuine')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
