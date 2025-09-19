# app.py
"""
Flask app that serves a simple web UI and a JSON /predict endpoint.
Run: python app.py  OR open in IDLE and Run Module (F5)
Then open http://127.0.0.1:5000 in your browser.
"""
import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Path to the model saved by train_task3.py
MODEL_PATH = os.path.join('models', 'iris_pipeline.joblib')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found. Train first (train_task3.py). Expected {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
pipeline = bundle['pipeline']
target_names = bundle['target_names']
feature_names_short = bundle['feature_names_short']  # ['sepal_length', ...]

@app.route('/')
def index():
    # Render a simple HTML form (templates/index.html)
    return render_template('index.html', feature_names_short=feature_names_short)

def json_to_feature_array(data):
    """
    Accepts either:
      {"features": [v0, v1, v2, v3]}
    or named keys:
      {"sepal_length": 5.1, "sepal_width": 3.5, ...}
    Returns numpy array shape (1,4)
    """
    if isinstance(data, dict) and 'features' in data:
        arr = np.array(data['features'], dtype=float).reshape(1, -1)
        return arr
    values = []
    for k in feature_names_short:
        if k in data:
            values.append(float(data[k]))
        else:
            # try fallback by short key without underscore
            if k.replace('_', '') in data:
                values.append(float(data[k.replace('_', '')]))
            else:
                raise ValueError(f"Missing key '{k}' in JSON payload")
    return np.array(values, dtype=float).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            X = json_to_feature_array(data)
        else:
            # form submission (from the HTML form)
            vals = []
            for k in feature_names_short:
                v = request.form.get(k)
                if v is None:
                    return jsonify({'error': f"Missing form field {k}"}), 400
                vals.append(float(v))
            X = np.array(vals).reshape(1, -1)

        pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X).max()
        pred_name = target_names[int(pred)]
        return jsonify({'prediction': pred_name, 'probability': float(proba)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Running with debug=True only for development/testing
    app.run(debug=True, host='127.0.0.1', port=5000)
