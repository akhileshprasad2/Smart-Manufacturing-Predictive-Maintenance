
import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load trained models and model information
with open(BASE_DIR / "models" / "ai4i_best_model.pkl", "rb") as f:
    ai4i_model = joblib.load(f)

with open(BASE_DIR / "models" / "secom_best_model.pkl", "rb") as f:
    secom_model = joblib.load(f)

with open(BASE_DIR / "models" / "model_summary.pkl", "rb") as f:
    model_summary = joblib.load(f)

ai4i_features = model_summary["AI4I"]["features"]
secom_features = model_summary["SECOM"]["features"]

# Load project metadata and comparison results
with open(BASE_DIR / "project_metadata.json", "r") as f:
    metadata = json.load(f)

ai4i_clean = [None] * metadata["ai4i_records"]
secom_final = [None] * metadata["secom_records"]

ai4i_results = pd.read_csv(
    BASE_DIR / "results" / "ai4i_model_comparison.csv"
)

secom_results = pd.read_csv(
    BASE_DIR / "results" / "secom_model_comparison.csv"
)

prediction_history = []


def get_machine_status(prediction, probability):

    if prediction == 1:
        return "Maintenance Required"

    if probability >= 0.30:
        return "Warning - Monitor Machine"

    return "Healthy"


def get_recommendation(prediction, probability):

    if prediction == 1:
        return (
            "Schedule machine inspection and maintenance. "
            "Check operating conditions and sensor readings."
        )

    if probability >= 0.30:
        return (
            "Monitor machine readings closely and schedule "
            "preventive inspection if risk increases."
        )

    return (
        "Machine appears healthy. Continue regular monitoring "
        "and preventive maintenance."
    )


def save_prediction(dataset, prediction, probability, status):

    prediction_history.append({
        "dataset": dataset,
        "prediction": int(prediction),
        "failure_probability": round(float(probability) * 100, 2),
        "status": status
    })


@app.route("/api/predict/ai4i", methods=["POST"])
def predict_ai4i():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please provide valid JSON sensor values."
            }), 400

        missing = [
            feature for feature in ai4i_features
            if feature not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing sensor values",
                "missing_features": missing
            }), 400

        input_df = pd.DataFrame(
            [[data[feature] for feature in ai4i_features]],
            columns=ai4i_features
        )

        # Convert numerical features
        for column in input_df.columns:
            if column != "Type":
                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="raise"
                )

        prediction = int(ai4i_model.predict(input_df)[0])

        probability = float(
            ai4i_model.predict_proba(input_df)[0][1]
        )

        status = get_machine_status(
            prediction,
            probability
        )

        recommendation = get_recommendation(
            prediction,
            probability
        )

        save_prediction(
            "AI4I",
            prediction,
            probability,
            status
        )

        return jsonify({
            "success": True,
            "dataset": "AI4I",
            "prediction": prediction,
            "prediction_label": (
                "Machine Failure" if prediction == 1
                else "Normal Operation"
            ),
            "failure_probability": round(
                probability * 100, 2
            ),
            "machine_status": status,
            "recommendation": recommendation,
            "model_used": model_summary["AI4I"]["best_model"]
        })

    except (ValueError, TypeError) as error:

        return jsonify({
            "error": f"Invalid sensor input: {str(error)}"
        }), 400

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


@app.route("/api/dashboard", methods=["GET"])
def dashboard():

    return jsonify({
        "project": "Big Data Analytics for Predictive Maintenance",
        "datasets": {
            "AI4I": {
                "records": len(ai4i_clean),
                "features": len(ai4i_features),
                "best_model": model_summary["AI4I"]["best_model"]
            },
            "SECOM": {
                "records": len(secom_final),
                "features": len(secom_features),
                "best_model": model_summary["SECOM"]["best_model"]
            }
        },
        "total_predictions": len(prediction_history),
        "total_failures_detected": sum(
            item["prediction"] == 1
            for item in prediction_history
        )
    })


@app.route("/api/history", methods=["GET"])
def history():

    return jsonify({
        "total": len(prediction_history),
        "history": prediction_history[-100:]
    })


@app.route("/api/model-comparison", methods=["GET"])
def model_comparison():

    comparison = pd.concat(
        [ai4i_results, secom_results],
        ignore_index=True
    )

    return jsonify(
        comparison.round(4).to_dict(orient="records")
    )


@app.route("/api/download/secom-results", methods=["GET"])
def download_secom_results():

    from flask import send_file

    path = BASE_DIR / "results" / "secom_predictions.csv"

    if not path.exists():
        return jsonify({"error": "No prediction results available."}), 404

    return send_file(
        path,
        as_attachment=True,
        download_name="secom_predictions.csv"
    )


@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "ai4i_model_loaded": ai4i_model is not None,
        "secom_model_loaded": secom_model is not None
    })


@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


def predict_secom_corrected():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Please upload a CSV file."}), 400

        file = request.files["file"]

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a CSV file."}), 400

        uploaded_df = pd.read_csv(file)

        # Keep all sensor column names as strings
        uploaded_df.columns = [
            str(col).strip() for col in uploaded_df.columns
        ]

        # Match the exact feature names expected by the trained model
        expected_features = [str(col) for col in secom_features]

        missing = [
            col for col in expected_features
            if col not in uploaded_df.columns
        ]

        if missing:
            return jsonify({
                "error": "Missing required SECOM columns.",
                "missing_count": len(missing),
                "missing_features": missing[:20]
            }), 400

        X_upload = uploaded_df.loc[:, expected_features].copy()

        predictions = np.asarray(secom_model.predict(X_upload))

        result_df = pd.DataFrame({
            "Row": range(1, len(predictions) + 1),
            "Prediction": predictions,
            "Machine Status": [
                "Failure" if int(p) == 1 else "Normal"
                for p in predictions
            ]
        })

        os.makedirs("results", exist_ok=True)

        result_df.to_csv(
            "results/secom_predictions.csv",
            index=False
        )

        return jsonify({
            "success": True,
            "message": "SECOM predictions generated successfully!",
            "total_rows": len(result_df),
            "failure_count": int((predictions == 1).sum()),
            "normal_count": int((predictions == 0).sum()),
            "predictions": result_df.to_dict(orient="records")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Register corrected SECOM prediction endpoint
app.add_url_rule(
    "/api/predict/secom",
    endpoint="predict_secom",
    view_func=predict_secom_corrected,
    methods=["POST"]
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
