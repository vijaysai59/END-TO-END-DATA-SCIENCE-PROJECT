# END-TO-END-DATA-SCIENCE-PROJECT











*DESCRIPTION*:

📌 Overview

This task demonstrates the full data-science pipeline:
data loading ➜ preprocessing ➜ model training ➜ model persistence ➜ deployment as a simple Flask web application.

The project uses the classic Iris dataset and exposes a web interface and a JSON API to predict the iris flower species from four numeric measurements.

✨ Features

Data Processing & Model Training

Loads the Iris dataset.

Scales features with StandardScaler.

Trains a RandomForestClassifier.

Saves the trained pipeline to models/iris_pipeline.joblib.

Web Application (Flask)

Single-page form to input sepal and petal measurements.

/predict endpoint returns JSON predictions and probability.

Interactive UI

Browser form posts JSON and displays the model’s output live.

🛠️ Project Structure
task3/
├─ app.py                  # Flask app serving predictions
├─ train_task3.py          # Script to train & save the model
├─ templates/
│   └─ index.html          # Simple HTML form
├─ models/
│   └─ iris_pipeline.joblib# Saved model pipeline
└─ requirements.txt        # Dependencies

🚀 Quick Start

Install dependencies

pip install -r requirements.txt


Train the model (creates models/iris_pipeline.joblib)

python train_task3.py


Run the web server

python app.py


Flask will start at http://127.0.0.1:5000.

Use the app

Open a browser to http://127.0.0.1:5000.

Enter sepal/petal length & width, click Predict.

View the JSON prediction and probability.

📂 Example Prediction

Input:

Sepal Length: 5.1
Sepal Width : 3.5
Petal Length: 1.4
Petal Width : 0.2


Response:

{
  "prediction": "setosa",
  "probability": 1.0
}
