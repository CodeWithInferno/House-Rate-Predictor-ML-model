# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from flask import Flask, request, jsonify
# import joblib

# # Flask app
# app = Flask(__name__)

# # Step 1: Train and Save the Model with New Dataset
# def train_and_save_model():
#     print("Loading dataset...")
#     df = pd.read_csv("American_Housing_Data_20231209.csv")

#     # Drop duplicates
#     df = df.drop_duplicates()

#     # Handle missing values (e.g., fill with median)
#     df.fillna(df.median(numeric_only=True), inplace=True)

#     # Select features and target
#     X = df[["Beds", "Baths", "Living Space", "Zip Code Population", "Median Household Income"]]
#     y = df["Price"]

#     # Split into training and testing sets
#     print("Splitting data...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create a preprocessing pipeline (e.g., scaling numerical features)
#     numeric_features = ["Beds", "Baths", "Living Space", "Zip Code Population", "Median Household Income"]
#     numeric_transformer = StandardScaler()

#     preprocessor = ColumnTransformer(transformers=[
#         ("num", numeric_transformer, numeric_features)
#     ])

#     # Define the model pipeline
#     model = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
#     ])

#     # Train the model
#     print("Training the model...")
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean Squared Error: {mse}\n")

#     # Save the model to a file
#     joblib.dump(model, "house_price_model_large.pkl")
#     print("Model trained and saved as 'house_price_model_large.pkl'!")

#     # Visualization
#     visualize_model(y_test, y_pred)

# def visualize_model(y_test, y_pred):
#     # Actual vs Predicted: Testing Data
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_pred, color="purple")
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Prediction")
#     plt.title("Actual vs Predicted (Testing Data)")
#     plt.xlabel("Actual Price")
#     plt.ylabel("Predicted Price")
#     plt.legend()
#     plt.show()

# train_and_save_model()

# # Step 2: Flask API to Use the Updated Model
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Load the saved model
#         model = joblib.load("house_price_model_large.pkl")

#         # Get input data from request
#         data = request.json
#         features = ["Beds", "Baths", "Living Space", "Zip Code Population", "Median Household Income"]
#         input_data = {feature: data.get(feature) for feature in features}

#         if None in input_data.values():
#             return jsonify({"error": f"Please provide all the required features: {features}"}), 400

#         # Convert input to DataFrame
#         new_data = pd.DataFrame([input_data])

#         # Make a prediction
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
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor  # Using Random Forest
# from flask import Flask, request, jsonify
# import joblib

# # Flask app
# app = Flask(__name__)

# # Step 1: Train and Save the Model with Improved Features and Algorithm
# def train_and_save_model():
#     print("Loading dataset...")
#     df = pd.read_csv("American_Housing_Data_20231209.csv")

#     # Drop duplicates
#     df = df.drop_duplicates()

#     # Handle missing values (e.g., fill with median)
#     df.fillna(df.median(numeric_only=True), inplace=True)

#     # Feature engineering
#     print("Engineering new features...")
#     # Add interaction terms (e.g., Baths per Bedroom)
#     df["Baths_per_Bedroom"] = df["Baths"] / (df["Beds"] + 1e-6)  # Avoid division by zero
#     # Add log-transformed population
#     df["Log_Zip_Code_Population"] = np.log1p(df["Zip Code Population"])
#     # Add Zip Code Density as is
#     df["Zip_Code_Density"] = df["Zip Code Density"]

#     # Select features and target
#     features = [
#         "Beds", 
#         "Baths", 
#         "Living Space", 
#         "Log_Zip_Code_Population", 
#         "Median Household Income", 
#         "Baths_per_Bedroom", 
#         "Zip_Code_Density", 
#         "Latitude", 
#         "Longitude"
#     ]
#     X = df[features]
#     y = df["Price"]

#     # Split into training and testing sets
#     print("Splitting data...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create a preprocessing pipeline
#     numeric_features = features
#     numeric_transformer = StandardScaler()

#     preprocessor = ColumnTransformer(transformers=[
#         ("num", numeric_transformer, numeric_features)
#     ])

#     # Define the model pipeline with Random Forest
#     model = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
#     ])

#     # Train the model
#     print("Training the model...")
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean Squared Error: {mse}\n")

#     # Save the model to a file
#     joblib.dump(model, "house_price_model_optimized.pkl")
#     print("Model trained and saved as 'house_price_model_optimized.pkl'!")

#     # Visualization
#     visualize_model(y_test, y_pred)

# def visualize_model(y_test, y_pred):
#     # Actual vs Predicted: Testing Data
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_pred, color="purple")
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Prediction")
#     plt.title("Actual vs Predicted (Testing Data)")
#     plt.xlabel("Actual Price")
#     plt.ylabel("Predicted Price")
#     plt.legend()
#     plt.show()

# train_and_save_model()

# # Step 2: Flask API to Use the Updated Model
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Load the saved model
#         model = joblib.load("house_price_model_optimized.pkl")

#         # Get input data from request
#         features = [
#             "Beds", 
#             "Baths", 
#             "Living Space", 
#             "Log_Zip_Code_Population", 
#             "Median Household Income", 
#             "Baths_per_Bedroom", 
#             "Zip_Code_Density", 
#             "Latitude", 
#             "Longitude"
#         ]
#         data = request.json
#         input_data = {feature: data.get(feature) for feature in features}

#         if None in input_data.values():
#             return jsonify({"error": f"Please provide all the required features: {features}"}), 400

#         # Convert input to DataFrame
#         new_data = pd.DataFrame([input_data])

#         # Make a prediction
#         prediction = model.predict(new_data)

#         return jsonify({"Predicted Price": prediction[0]})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)













# import os
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify, render_template
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import joblib

# # Flask app
# app = Flask(__name__)

# MODEL_FILE = "house_price_model_optimized.pkl"

# # Check if the model exists
# if not os.path.exists(MODEL_FILE):
#     def train_and_save_model():
#         print("Loading dataset...")
#         df = pd.read_csv("American_Housing_Data_20231209.csv")

#         # Drop duplicates
#         df = df.drop_duplicates()

#         # Handle missing values
#         df.fillna(df.median(numeric_only=True), inplace=True)

#         # Feature engineering
#         print("Engineering new features...")
#         df["Baths_per_Bedroom"] = df["Baths"] / (df["Beds"] + 1e-6)
#         df["Log_Zip_Code_Population"] = np.log1p(df["Zip Code Population"])
#         df["Zip_Code_Density"] = df["Zip Code Density"]

#         # Select features and target
#         features = [
#             "Beds", 
#             "Baths", 
#             "Living Space", 
#             "Log_Zip_Code_Population", 
#             "Median Household Income", 
#             "Baths_per_Bedroom", 
#             "Zip_Code_Density", 
#             "Latitude", 
#             "Longitude"
#         ]
#         X = df[features]
#         y = df["Price"]

#         # Split into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Create a preprocessing pipeline
#         numeric_features = features
#         numeric_transformer = StandardScaler()

#         preprocessor = ColumnTransformer(transformers=[
#             ("num", numeric_transformer, numeric_features)
#         ])

#         # Define the model pipeline with Random Forest
#         model = Pipeline(steps=[
#             ("preprocessor", preprocessor),
#             ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
#         ])

#         # Train the model
#         print("Training the model...")
#         model.fit(X_train, y_train)

#         # Evaluate the model
#         y_pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#         print(f"Mean Squared Error: {mse}\n")

#         # Save the model to a file
#         joblib.dump(model, MODEL_FILE)
#         print(f"Model trained and saved as '{MODEL_FILE}'!")

#     # Train and save the model if it doesn't exist
#     train_and_save_model()

# # Flask Routes
# @app.route("/")
# def home():
#     return render_template("index.html")  # Render the input form

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Load the saved model
#         model = joblib.load(MODEL_FILE)

#         # Get input data from form
#         features = [
#             "Beds", 
#             "Baths", 
#             "Living Space", 
#             "Log_Zip_Code_Population", 
#             "Median Household Income", 
#             "Baths_per_Bedroom", 
#             "Zip_Code_Density", 
#             "Latitude", 
#             "Longitude"
#         ]
#         input_data = {feature: request.form.get(feature, type=float) for feature in features}

#         if None in input_data.values():
#             return jsonify({"error": "Please provide valid values for all fields."}), 400

#         # Convert input to DataFrame
#         new_data = pd.DataFrame([input_data])

#         # Make a prediction
#         prediction = model.predict(new_data)

#         return render_template("index.html", prediction_text=f"Predicted Price: ${prediction[0]:,.2f}")

#     except Exception as e:
#         return render_template("index.html", prediction_text=f"Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)





















import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Flask app
app = Flask(__name__)

MODEL_FILE = "house_price_model_optimized.pkl"

# Visualization functions
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='purple', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig("actual_vs_predicted.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.named_steps["regressor"].feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance, color='skyblue')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig("feature_importance.png")
    plt.close()
def plot_correlation_heatmap(data):
    """Plot a correlation heatmap for numeric columns only."""
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Compute the correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved as 'correlation_heatmap.png'.")
    plt.close()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='blue')
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title('Residuals')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.savefig("residuals.png")
    plt.close()

# Check if the model exists
if not os.path.exists(MODEL_FILE):
    def train_and_save_model():
        print("Loading dataset...")
        df = pd.read_csv("American_Housing_Data_20231209.csv")

        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Feature engineering
        print("Engineering new features...")
        df["Baths_per_Bedroom"] = df["Baths"] / (df["Beds"] + 1e-6)
        df["Log_Zip_Code_Population"] = np.log1p(df["Zip Code Population"])
        df["Zip_Code_Density"] = df["Zip Code Density"]

        # Select features and target
        features = [
            "Beds", 
            "Baths", 
            "Living Space", 
            "Log_Zip_Code_Population", 
            "Median Household Income", 
            "Baths_per_Bedroom", 
            "Zip_Code_Density", 
            "Latitude", 
            "Longitude"
        ]
        X = df[features]
        y = df["Price"]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a preprocessing pipeline
        numeric_features = features
        numeric_transformer = StandardScaler()

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features)
        ])

        # Define the model pipeline with Random Forest
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ])

        # Train the model
        print("Training the model...")
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}\n")

        # Save the model to a file
        joblib.dump(model, MODEL_FILE)
        print(f"Model trained and saved as '{MODEL_FILE}'!")

        # Generate and save visualizations
        print("Generating visualizations...")
        plot_actual_vs_predicted(y_test, y_pred)
        plot_feature_importance(model, features)
        plot_correlation_heatmap(df)
        plot_residuals(y_test, y_pred)

    # Train and save the model if it doesn't exist
    train_and_save_model()

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")  # Render the input form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the saved model
        model = joblib.load(MODEL_FILE)

        # Get input data from form
        features = [
            "Beds", 
            "Baths", 
            "Living Space", 
            "Log_Zip_Code_Population", 
            "Median Household Income", 
            "Baths_per_Bedroom", 
            "Zip_Code_Density", 
            "Latitude", 
            "Longitude"
        ]
        input_data = {feature: request.form.get(feature, type=float) for feature in features}

        if None in input_data.values():
            return jsonify({"error": "Please provide valid values for all fields."}), 400

        # Convert input to DataFrame
        new_data = pd.DataFrame([input_data])

        # Make a prediction
        prediction = model.predict(new_data)

        return render_template("index.html", prediction_text=f"Predicted Price: ${prediction[0]:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
