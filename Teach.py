import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
import joblib

# Flask app
app = Flask(__name__)

# Step 1: Load and Prepare the Dataset
def load_and_prepare_data():
    print("Loading dataset...")
    df = pd.read_csv("House Data Large.csv")  # Ensure this file exists

    # Features (X) and Target (y)
    X = df[["Area", "Rooms"]]
    y = df["Price"]

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Step 2: Train the Model
def train_model(X_train, y_train):
    print("Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete!")

    return model

# Step 3: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}\n")

    return mse

# Step 4: Save the Model
def save_model(model):
    print("Saving the trained model...")
    joblib.dump(model, "house_price_model.pkl")
    print("Model saved as 'house_price_model.pkl'!")

# Step 5: Flask API for Predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Loading the saved model...")
        model = joblib.load("house_price_model.pkl")

        # Get input data from request
        data = request.json
        area = data.get("Area")
        rooms = data.get("Rooms")

        if area is None or rooms is None:
            return jsonify({"error": "Please provide both 'Area' and 'Rooms' in the request."}), 400

        # Make a prediction
        new_data = pd.DataFrame({"Area": [area], "Rooms": [rooms]})
        prediction = model.predict(new_data)

        return jsonify({"Predicted Price": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main Function
def main():
    # Step 1: Load and Prepare Data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Step 2: Train the Model
    model = train_model(X_train, y_train)

    # Step 3: Evaluate the Model
    evaluate_model(model, X_test, y_test)

    # Step 4: Save the Model
    save_model(model)

if __name__ == "__main__":
    main()
    app.run(debug=True)
