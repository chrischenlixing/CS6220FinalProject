from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Create a Flask application
app = Flask(__name__)

# Load the pre-trained model
model_filename = "random_forest_model.pkl"
try:
    model = joblib.load(model_filename)
    print(f"Model loaded successfully from {model_filename}")
except FileNotFoundError:
    raise Exception(f"Model file '{model_filename}' not found. Ensure it is in the application directory.")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_user_conversion():
    """
    Handle POST requests for user conversion predictions.
    Expects JSON input containing the features required by the model.
    """
    # Get the JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data])

        # Check for missing features required by the model
        required_features = model.feature_names_in_
        print(f"Required features: {required_features}")
        missing_features = [feature for feature in required_features if feature not in input_data.columns]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # Reorder columns to match the model's feature order
        input_data = input_data[required_features]

        # Make predictions using the model
        prediction = model.predict(input_data)[0]
        # Calculate confidence based on majority voting from decision trees
        votes = [tree.predict(input_data)[0] for tree in model.estimators_]
        confidence = votes.count(prediction) / len(votes)

        # Return the prediction and confidence
        response = {
            'prediction': int(prediction),
            'confidence': round(float(confidence), 2)
        }
        return jsonify(response)

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Start the Flask application
if __name__ == "__main__":
    # Get the port from environment variables (default to 8080)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
