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
    pclass = int(request.form['Pclass'])
    sex = int(request.form['Sex'])
    age = float(request.form['Age'])
    sibsp = int(request.form['SibSp'])
    parch = int(request.form['Parch'])
    fare = float(request.form['Fare'])
    embarked = int(request.form['Embarked'])
    
    # Create input array for the model
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    
    # Predict using the model
    prediction = model.predict(features)
    
    # Interpret the result
    output = 'Survived' if prediction[0] == 1 else 'Not Survived'
    
    return render_template('index.html', prediction_text=f'Passenger would have: {output}')

if __name__ == "__main__":
    app.run(debug=True)
