import flask
from flask import request, render_template
import joblib
import pandas as pd
import numpy as np
import json

app = flask.Flask(__name__)

try:
    model = joblib.load('api/model.pkl')
    feature_names = joblib.load('api/feature_names.pkl')
    with open('api/model_performance.json', 'r') as f:
        performance_data = json.load(f)
    print("Model, fitur, dan data kinerja berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan ({e}). Jalankan 'train_model.py' terlebih dahulu.")
    exit()

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html', 
                               features=feature_names, 
                               performance=performance_data)

    if request.method == 'POST':
        input_features = {}
        for feature in feature_names:
            value = request.form.get(feature)
            try:
                input_features[feature] = float(value)
            except (ValueError, TypeError):
                return render_template('index.html', 
                                       features=feature_names, 
                                       performance=performance_data,
                                       error=f"Input untuk '{feature}' tidak valid. Harap masukkan angka.")

        input_df = pd.DataFrame([input_features], columns=feature_names)
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', 
                               features=feature_names, 
                               prediction=prediction,
                               input_values=input_features,
                               performance=performance_data)

if __name__ == '__main__':
    app.run(debug=True)