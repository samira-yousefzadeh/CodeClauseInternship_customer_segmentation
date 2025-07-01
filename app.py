import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get data from form
            age = float(request.form["age"])
            income = float(request.form["income"])
            spending = float(request.form["spending"])

            # Scale the input
            input_data = scaler.transform([[age, income, spending]])

            # Predict the cluster
            cluster = model.predict(input_data)[0]
            prediction = f"The customer belongs to Cluster {cluster}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
