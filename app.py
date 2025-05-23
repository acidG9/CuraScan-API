
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load your trained model and scaler at startup (once only)
data = joblib.load("trained_models_and_scaler.joblib")
model = data["rf_model"]
scaler = data["scaler"]
feature_names = scaler.feature_names_in_

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    print("✅ Server is running. Use POST /predict to get predictions.")
    return jsonify({"message": "✅ Server is running. Use POST /predict to get predictions."})

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON from request (raise error if not present)
    input_json = request.get_json(force=True)
    # Ensure the data fields map correctly to your feature names and order
    try:
        df = pd.DataFrame([input_json], columns=feature_names)
        df.columns = feature_names  # Make sure columns are matched exactly
        scaled = scaler.transform(df)
        prediction = int(model.predict(scaled)[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

