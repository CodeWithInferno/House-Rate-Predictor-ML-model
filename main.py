# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from tqdm import tqdm  # For the loading bar
# import time  # Simulate delays for demonstration

# # Load the dataset from CSV
# print("Loading dataset...")
# df = pd.read_csv("House Data Large.csv")  # Adjusted for the correct file name

# # Features (X) and Target (y)
# X = df[["Area", "Rooms"]]
# y = df["Price"]

# # Split into training and testing sets
# print("Splitting data...")
# time.sleep(1)  # Simulate delay
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the model
# print("Training the model...")
# for _ in tqdm(range(100), desc="Training Progress"):
#     time.sleep(0.01)  # Simulate delay for loading bar
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on the test set
# print("\nMaking predictions...")
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")

# # Feedback: Show predictions vs actual values (first 10 for brevity)
# print("Predictions vs Actual Values (First 10):")
# for actual, predicted in zip(y_test.values[:10], y_pred[:10]):
#     print(f"Actual: {actual}, Predicted: {predicted:.2f}")












# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from flask import Flask, request, jsonify
# import joblib

# # Flask app
# app = Flask(__name__)

# # Step 1: Train and Save the Model
# def train_and_save_model():
#     print("Loading dataset...")
#     df = pd.read_csv("House Data Large.csv")  # Make sure this file exists

#     # Features (X) and Target (y)
#     X = df[["Area", "Rooms"]]
#     y = df["Price"]

#     # Split into training and testing sets
#     print("Splitting data...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train the model
#     print("Training the model...")
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean Squared Error: {mse}\n")

#     # Save the model to a file
#     joblib.dump(model, "house_price_model.pkl")
#     print("Model trained and saved as 'house_price_model.pkl'!")

# train_and_save_model()

# # Step 2: Flask API to Use the Model
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Load the saved model
#         model = joblib.load("house_price_model.pkl")

#         # Get input data from request
#         data = request.json
#         area = data.get("Area")
#         rooms = data.get("Rooms")

#         if area is None or rooms is None:
#             return jsonify({"error": "Please provide both 'Area' and 'Rooms' in the request."}), 400

#         # Make a prediction
#         new_data = pd.DataFrame({"Area": [area], "Rooms": [rooms]})
#         prediction = model.predict(new_data)

#         return jsonify({"Predicted Price": prediction[0]})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)





















# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from flask import Flask, request, jsonify
# import joblib

# # Flask app
# app = Flask(__name__)

# # Step 1: Train and Save the Model with Visualization
# def train_and_save_model():
#     print("Loading dataset...")
#     df = pd.read_csv("House Data Large.csv")  # Make sure this file exists

#     # Features (X) and Target (y)
#     X = df[["Area", "Rooms"]]
#     y = df["Price"]

#     # Split into training and testing sets
#     print("Splitting data...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train the model
#     print("Training the model...")
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean Squared Error: {mse}\n")

#     # Save the model to a file
#     joblib.dump(model, "house_price_model.pkl")
#     print("Model trained and saved as 'house_price_model.pkl'!")

#     # Visualization
#     visualize_model(X_train, y_train, X_test, y_test, model)

# def visualize_model(X_train, y_train, X_test, y_test, model):
#     # Predictions for visualization
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)

#     # 1. Scatter Plot: Area vs. Price
#     plt.figure(figsize=(10, 6))
#     plt.scatter(X_train["Area"], y_train, color="blue", label="Training Data")
#     plt.scatter(X_test["Area"], y_test, color="orange", label="Testing Data")
#     plt.plot(X_train["Area"], y_train_pred, color="red", label="Regression Line (Train)")
#     plt.title("Area vs Price")
#     plt.xlabel("Area")
#     plt.ylabel("Price")
#     plt.legend()
#     plt.show()

#     # 2. Residual Plot: Actual vs Predicted
#     plt.figure(figsize=(10, 6))
#     residuals = y_test - y_test_pred
#     plt.scatter(y_test_pred, residuals, color="green")
#     plt.axhline(0, color="red", linestyle="--")
#     plt.title("Residual Plot (Actual - Predicted)")
#     plt.xlabel("Predicted Price")
#     plt.ylabel("Residuals")
#     plt.show()

#     # 3. Actual vs Predicted: Testing Data
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_test_pred, color="purple")
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Prediction")
#     plt.title("Actual vs Predicted (Testing Data)")
#     plt.xlabel("Actual Price")
#     plt.ylabel("Predicted Price")
#     plt.legend()
#     plt.show()

# train_and_save_model()

# # Step 2: Flask API to Use the Model
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Load the saved model
#         model = joblib.load("house_price_model.pkl")

#         # Get input data from request
#         data = request.json
#         area = data.get("Area")
#         rooms = data.get("Rooms")

#         if area is None or rooms is None:
#             return jsonify({"error": "Please provide both 'Area' and 'Rooms' in the request."}), 400

#         # Make a prediction
#         new_data = pd.DataFrame({"Area": [area], "Rooms": [rooms]})
#         prediction = model.predict(new_data)

#         return jsonify({"Predicted Price": prediction[0]})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
import joblib

# Flask app
app = Flask(__name__)

# Step 1: Train and Save the Model with New Dataset
def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_csv("American_Housing_Data_20231209.csv")

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values (e.g., fill with median)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Select features and target
    X = df[["Beds", "Baths", "Living Space", "Zip Code Population", "Median Household Income"]]
    y = df["Price"]

    # Split into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a preprocessing pipeline (e.g., scaling numerical features)
    numeric_features = ["Beds", "Baths", "Living Space", "Zip Code Population", "Median Household Income"]
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features)
    ])

    # Define the model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}\n")

    # Save the model to a file
    joblib.dump(model, "house_price_model_large.pkl")
    print("Model trained and saved as 'house_price_model_large.pkl'!")

    # Visualization
    visualize_model(y_test, y_pred)

def visualize_model(y_test, y_pred):
    # Actual vs Predicted: Testing Data
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="purple")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Prediction")
    plt.title("Actual vs Predicted (Testing Data)")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.show()

train_and_save_model()

# Step 2: Flask API to Use the Updated Model
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the saved model
        model = joblib.load("house_price_model_large.pkl")

        # Get input data from request
        data = request.json
        features = ["Beds", "Baths", "Living Space", "Zip Code Population", "Median Household Income"]
        input_data = {feature: data.get(feature) for feature in features}

        if None in input_data.values():
            return jsonify({"error": f"Please provide all the required features: {features}"}), 400

        # Convert input to DataFrame
        new_data = pd.DataFrame([input_data])

        # Make a prediction
        prediction = model.predict(new_data)

        return jsonify({"Predicted Price": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
