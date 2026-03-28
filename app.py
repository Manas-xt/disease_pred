from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and symptom columns
model = joblib.load('disease_model.pkl')
symptom_cols = joblib.load('symptom_cols.pkl')

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptom_cols)

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])

    # Build input vector
    input_vector = [1 if col in selected_symptoms else 0 for col in symptom_cols]
    input_array = np.array(input_vector).reshape(1, -1)

    # Predict
    prediction = model.predict(input_array)[0]

    # Get top 3 probabilities if available
    try:
        proba = model.predict_proba(input_array)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {"disease": model.classes_[i], "confidence": round(float(proba[i]) * 100, 1)}
            for i in top3_idx if proba[i] > 0
        ]
    except:
        top3 = [{"disease": prediction, "confidence": 100.0}]

    return jsonify({
        "prediction": prediction,
        "top3": top3
    })

if __name__ == '__main__':
    app.run(debug=True)
