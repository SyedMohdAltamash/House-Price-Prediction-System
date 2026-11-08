import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

# Valid ocean proximity categories
VALID_OCEAN_PROXIMITY = ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]

def load_or_train_model():
    """Loads the trained model pipeline or trains it if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Training a new model...")
        from train_model import train_and_save_model
        return train_and_save_model()
    
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_or_train_model()

def validate_and_parse_inputs(source_data):
    """
    Validates and extracts house features from form or json dictionary.
    Returns (cleaned_data_dict, error_message).
    """
    fields = [
        ("longitude", float, "Longitude must be a valid number (e.g. -122.23)"),
        ("latitude", float, "Latitude must be a valid number (e.g. 37.88)"),
        ("housing_median_age", float, "Housing Age must be a positive number"),
        ("total_rooms", float, "Total Rooms must be a positive number"),
        ("total_bedrooms", float, "Total Bedrooms must be a positive number"),
        ("population", float, "Population must be a positive number"),
        ("households", float, "Households must be a positive number"),
        ("median_income", float, "Median Income must be a positive number (in $10k units, e.g. 3.5)"),
    ]

    cleaned = {}
    for name, cast_type, err_msg in fields:
        raw_val = source_data.get(name)
        if raw_val is None or str(raw_val).strip() == "":
            return None, f"Missing required field: {name.replace('_', ' ').title()}"
        try:
            val = cast_type(raw_val)
        except (ValueError, TypeError):
            return None, err_msg

        if name not in ("longitude", "latitude") and val < 0:
            return None, f"{name.replace('_', ' ').title()} cannot be negative"
        
        cleaned[name] = [val]

    ocean = source_data.get("ocean_proximity", "").strip()
    if ocean not in VALID_OCEAN_PROXIMITY:
        return None, f"Invalid Ocean Proximity option: {ocean}"
    cleaned["ocean_proximity"] = [ocean]

    return cleaned, None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    formatted_prediction = None
    error = None
    form_data = {
        "longitude": "",
        "latitude": "",
        "housing_median_age": "",
        "total_rooms": "",
        "total_bedrooms": "",
        "population": "",
        "households": "",
        "median_income": "",
        "ocean_proximity": "INLAND"
    }

    if request.method == "POST":
        # Keep submitted form values to preserve UI state
        for key in form_data.keys():
            if key in request.form:
                form_data[key] = request.form[key]

        data, err = validate_and_parse_inputs(request.form)
        if err:
            error = err
        else:
            try:
                input_df = pd.DataFrame(data)
                pred_val = float(model.predict(input_df)[0])
                prediction = round(pred_val, 2)
                formatted_prediction = f"${prediction:,.2f}"
            except Exception as e:
                error = f"Error during prediction: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        formatted_prediction=formatted_prediction,
        error=error,
        form_data=form_data,
        ocean_options=VALID_OCEAN_PROXIMITY
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic predictions."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    data, err = validate_and_parse_inputs(request.get_json())
    if err:
        return jsonify({"success": False, "error": err}), 400

    try:
        input_df = pd.DataFrame(data)
        pred_val = float(model.predict(input_df)[0])
        return jsonify({
            "success": True,
            "prediction": round(pred_val, 2),
            "formatted_prediction": f"${pred_val:,.2f}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")
    print(f"Starting House Price Prediction server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
