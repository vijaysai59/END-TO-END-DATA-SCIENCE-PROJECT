# train_task3.py
"""
Train an ML pipeline on the Iris dataset and save it to models/iris_pipeline.joblib.
Run this in IDLE: Run -> Run Module (F5)
"""
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names  # human readable
    # short names we'll use in the web form / JSON keys:
    feature_names_short = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build pipeline (scaler + classifier)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Fit
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6. Save model bundle
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'iris_pipeline.joblib')
    joblib.dump({
        'pipeline': pipeline,
        'target_names': list(target_names),
        'feature_names': feature_names,
        'feature_names_short': feature_names_short
    }, model_path)
    print(f"\nSaved model bundle to: {model_path}")

if __name__ == '__main__':
    main()
