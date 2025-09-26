Disease Prediction System

A machine learning–based desktop application built with Python, Scikit-learn, and Tkinter.
It predicts the likelihood of having Diabetes, Fever, or Heart Disease based on user input data.

🚀 Features

Trains models using RandomForestClassifier

Handles missing/zero values intelligently (for Diabetes dataset)

Saves trained models with Joblib (includes model, scaler, and feature names)

GUI built with Tkinter for easy interaction

Dynamic input fields generated based on disease type

Probability output for predictions

Error handling for missing files or invalid inputs

 Project Structure
Disease-Prediction-System/
│
├── diabetes.csv               # Dataset for diabetes
├── Fever-1.csv                 # Dataset for fever
├── heart_disease_data.csv      # Dataset for heart disease
├── disease_predictor.py        # Main code (training + GUI)
├── diabetes_model.pkl          # Trained diabetes model
├── fever_model.pkl             # Trained fever model
├── heart_model.pkl             # Trained heart disease model
└── README.md                   # Project documentation
