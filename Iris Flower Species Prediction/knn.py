from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

dataset_path = 'iris.csv'  

if os.path.exists(dataset_path):
    import pandas as pd
    data = pd.read_csv(dataset_path)

    label_encoder = LabelEncoder()
    label_encoder.fit(data['Species'].values)  
else:
    print(f"Dataset file '{dataset_path}' not found!")

model_filename = 'knn_model.joblib'

if os.path.exists(model_filename):
    knn = joblib.load(model_filename)
    print("Model loaded from file.")
else:
    print(f"Model file '{model_filename}' not found!")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = knn.predict(input_features)

    species = label_encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction=species)

if __name__ == "__main__":
    app.run(debug=True)
