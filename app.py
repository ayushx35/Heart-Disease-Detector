from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the app
app = Flask(__name__)

# Load and prepare the data
heart_data = pd.read_csv('data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Route for frontend
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input data to a numpy array
    input_data = [
        int(data['age']),
        int(data['sex']),
        int(data['cp']),
        int(data['trestbps']),
        int(data['chol']),
        int(data['fbs']),
        int(data['restecg']),
        int(data['thalach']),
        int(data['exang']),
        float(data['oldpeak']),
        int(data['slope']),
        int(data['ca']),
        int(data['thal'])
    ]
    
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_numpy_array)

    # Respond with prediction result
    if prediction[0] == 0:
        return jsonify({'message': 'The Person does not have Heart Disease'})
    else:
        return jsonify({'message': 'The Person has Heart Disease'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
