import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
import joblib

# Flask app
app = Flask(__name__)

# Step 1: Load and Preprocess the Dataset
def load_and_preprocess():
    # Load the dataset
    df = pd.read_csv("student-mat.csv", sep=";")

    # Select relevant features and target
    features = ["G1", "G2", "studytime", "failures", "absences"]
    target = "G3"
    X = df[features]
    y = df[target]

    return X, y

# Step 2: Train and Save the Model
def train_and_save_model():
    X, y = load_and_preprocess()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Save the model
    joblib.dump(model, "grade_predictor.pkl")
    print("Model saved as 'grade_predictor.pkl'!")

train_and_save_model()

# Step 3: Flask API for Grade Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the saved model
        model = joblib.load("grade_predictor.pkl")

        # Get input data from the request
        data = request.json
        input_features = ["G1", "G2", "studytime", "failures", "absences"]
        input_data = pd.DataFrame([data], columns=input_features)

        # Make a prediction
        prediction = model.predict(input_data)
        return jsonify({"Predicted Grade": round(prediction[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
