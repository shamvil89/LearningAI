import tkinter as tk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# Initialize the logistic regression model
model = LogisticRegression(tol=1e-4)

# Initialize lists to store the input data and corresponding labels
data = []
labels = []

# Initialize polynomial features
polynomial_features = PolynomialFeatures(degree=2)

# Function to collect the input data and labels
def collect_data(event=None):
    try:
        entry = int(entry_field.get())
        if entry < 1 or entry > 10:
            raise ValueError("Number must be between 1 and 10")

        error_label.config(text="")  # Clear error label
        data.append([entry])
        labels.append(entry)
        entry_field.delete(0, tk.END)

        counter_label.config(text="Entries Collected: {}".format(len(data)))

        remaining_entries = 10 - len(data)
        add_entry_button.config(text="Add {} Entries to train model".format(remaining_entries))

        if len(data) == 10:
            train_model()
            predict_number()
            entry_field.configure(state="disabled")  # Disable entry field
            actual_entry.configure(state="normal")  # Enable actual number entry
            compare_button.configure(state="normal")  # Enable compare button
            start_over_button.configure(state="normal")  # Enable start over button
            add_entry_button.configure(state="disabled")  # Disable add entry button
    except ValueError as e:
        error_label.config(text=str(e))

# Function to train the model
def train_model():
    X = np.array(data[:-1])  # Use the pattern of numbers as features
    y = np.array(labels[1:])  # Use the subsequent number as the label

    X_poly = polynomial_features.fit_transform(X)

    model.fit(X_poly, y)

# Function to predict a number based on the last entered value
def predict_number():
    entry = data[-1][0]
    X = np.array([[entry]])
    X_poly = polynomial_features.transform(X)
    predicted_number = model.predict(X_poly)
    predicted_label.config(text="Predicted Number: {}".format(predicted_number[0]))

# Function to compare the predicted number with the actual number
def compare_numbers():
    actual_number = int(actual_entry.get())
    predicted_number = int(predicted_label.cget("text").split(":")[1].strip())
    if actual_number == predicted_number:
        result_label.config(text="Result: Correct")
    else:
        result_label.config(text="Result: Incorrect")
        correct_data(actual_number)

# Function to correct the data and retrain the model
def correct_data(actual_number):
    data[-1][0] = actual_number
    train_model()

# Function to start over and reset the GUI
def start_over():
    global data, labels
    data = []
    labels = []
    entry_field.configure(state="normal")  # Enable entry field
    actual_entry.delete(0, tk.END)
    actual_entry.configure(state="disabled")  # Disable actual number entry
    compare_button.configure(state="disabled")  # Disable compare button
    predicted_label.config(text="Predicted Number: -")
    result_label.config(text="Result: -")
    counter_label.config(text="Entries Collected: 0")
    start_over_button.configure(state="disabled")  # Disable start over button
    error_label.config(text="")  # Reset error label
    add_entry_button.configure(state="normal", text="Add 10 Entries to train model")

# Create the main window
window = tk.Tk()
window.title("Number Predictor")

# Create the top label
top_label = tk.Label(window, text="Enter a number between 1 and 10", font=("Arial", 12))
top_label.pack(pady=10)

# Create and configure the entry field
entry_field = tk.Entry(window, width=20)
entry_field.pack(pady=10)
entry_field.bind("<Return>", collect_data)
entry_field.focus_set()

# Create the add entry button
add_entry_button = tk.Button(window, text="Add 10 Entries to train model", command=collect_data)
add_entry_button.pack(pady=5)

# Create the counter label
counter_label = tk.Label(window, text="Entries Collected: 0", font=("Arial", 12))
counter_label.pack(pady=10)

# Create the predicted label
predicted_label = tk.Label(window, text="Predicted Number: -", font=("Arial", 14))
predicted_label.pack(pady=10)

# Create the actual number entry and label
actual_frame = tk.Frame(window)
actual_frame.pack(pady=5)
actual_label = tk.Label(actual_frame, text="Enter Actual Number:", font=("Arial", 12))
actual_label.pack(side="left")
actual_entry = tk.Entry(actual_frame, width=10, state="disabled")
actual_entry.pack(side="left")

# Create the compare button
compare_button = tk.Button(window, text="Compare", command=compare_numbers, state="disabled")
compare_button.pack(pady=5)

# Create the result label
result_label = tk.Label(window, text="Result: -", font=("Arial", 12))
result_label.pack(pady=10)

# Create the error label
error_label = tk.Label(window, text="", font=("Arial", 12), fg="red")
error_label.pack(pady=5)

# Create the start over button
start_over_button = tk.Button(window, text="Start Over", command=start_over, state="disabled")
start_over_button.pack(pady=5)

# Start the GUI event loop
window.mainloop()
