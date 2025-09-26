import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk, messagebox


# ------------------- MODEL TRAINING -------------------
def train_and_save_model(file, target_col, model_name, replace_zero_cols=None):
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        print(f"⚠️ File {file} not found. Skipping {model_name}.")
        return

    # ✅ Check if target column exists
    if target_col not in data.columns:
        print(f"⚠️ Target column '{target_col}' not found in {file}.")
        print(f"👉 Available columns: {list(data.columns)}")
        return

    # Replace 0 with NaN and fill with mean (only for specified columns)
    if replace_zero_cols:
        data[replace_zero_cols] = data[replace_zero_cols].replace(0, np.nan)
        data.fillna(data.mean(), inplace=True)

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{model_name} trained ✅ Accuracy: {acc:.3f}")

    # Save model, scaler, and feature names
    saved = {"model": model, "scaler": scaler, "features": X.columns.tolist()}
    joblib.dump(saved, model_name, compress=3)


# Train models with correct file names
train_and_save_model(
    "diabetes.csv", "Outcome", "diabetes_model.pkl",
    replace_zero_cols=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
)

train_and_save_model("Fever-1.csv", "Outcome", "fever_model.pkl")

# ⚠️ If 'Outcome' is wrong in your CSV, check printed columns and change target_col
train_and_save_model("heart_disease_data.csv", "Outcome", "heart_model.pkl")


# ------------------- GUI -------------------
def launch_gui():
    root = tk.Tk()
    root.title("Disease Prediction System")
    root.geometry("700x700")

    # Dropdown for disease choice
    tk.Label(root, text="Select Disease:", font=("Arial", 12)).pack(pady=5)
    disease_choice = ttk.Combobox(root, values=["Diabetes", "Fever", "Heart Disease"], state="readonly")
    disease_choice.pack(pady=5)

    # Frame for feature inputs
    input_frame = tk.Frame(root)
    input_frame.pack(pady=10)
    entries = {}

    def load_inputs():
        for widget in input_frame.winfo_children():
            widget.destroy()
        entries.clear()

        choice = disease_choice.get()
        if choice == "":
            return

        models = {
            "Diabetes": "diabetes_model.pkl",
            "Fever": "fever_model.pkl",
            "Heart Disease": "heart_model.pkl"
        }

        model_file = models.get(choice)
        try:
            saved = joblib.load(model_file)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file {model_file} not found.")
            return

        features = saved["features"]

        tk.Label(input_frame, text=f"Enter values for {choice} features:",
                 font=("Arial", 12, "bold")).pack(pady=5)
        for f in features:
            frame = tk.Frame(input_frame)
            frame.pack(pady=2)
            tk.Label(frame, text=f"{f}: ", width=20, anchor="w").pack(side="left")
            ent = tk.Entry(frame)
            ent.pack(side="left")
            entries[f] = ent

    def predict_disease():
        choice = disease_choice.get()
        models = {
            "Diabetes": "diabetes_model.pkl",
            "Fever": "fever_model.pkl",
            "Heart Disease": "heart_model.pkl"
        }
        model_file = models.get(choice)

        if not model_file:
            messagebox.showerror("Error", "Please select a disease type.")
            return

        try:
            saved = joblib.load(model_file)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file {model_file} not found.")
            return

        model = saved["model"]
        scaler = saved["scaler"]
        features = saved["features"]

        # Validate inputs
        if any(entries[f].get().strip() == "" for f in features):
            messagebox.showerror("Error", "All fields must be filled.")
            return

        try:
            user_input = [float(entries[f].get()) for f in features]
        except:
            messagebox.showerror("Error", "Please enter valid numbers for all fields.")
            return

        user_array = np.array(user_input).reshape(1, -1)
        user_array = scaler.transform(user_array)
        pred = model.predict(user_array)[0]
        prob = model.predict_proba(user_array)[0][1]  # probability of disease

        if pred == 1:
            result_label.config(text=f"⚠️ Disease Detected! (Prob: {prob:.2%})", fg="red")
        else:
            result_label.config(text=f"✅ No Disease (Prob: {prob:.2%})", fg="green")

    # Buttons
    tk.Button(root, text="Load Features", command=load_inputs,
              bg="gray", fg="white").pack(pady=5)
    tk.Button(root, text="Predict", command=predict_disease,
              bg="blue", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

    global result_label
    result_label = tk.Label(root, text="", font=("Arial", 14))
    result_label.pack(pady=20)

    root.mainloop()


# ------------------- MAIN -------------------
if __name__ == "__main__":
    launch_gui()
