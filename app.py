import numpy as np
from flask import Flask, request, render_template
import pickle
import os

# Create a Flask web application instance
app = Flask(__name__)

# --- Load the Trained Model ---
# The model is loaded once when the application starts.
# We check if the model file exists before trying to load it.
model_file = 'model.pkl'
if not os.path.exists(model_file):
    print(f"Error: Model file '{model_file}' not found.")
    print("Please run the 'create_model.py' script first to train and save the model.")
    # The application will exit if the model is not found.
    exit()

# Load the pre-trained machine learning model from the file.
model = pickle.load(open(model_file, "rb"))

# --- Define Routes ---

# Define the route for the home page.
@app.route("/")
def home():
    """Renders the main page of the web application."""
    return render_template("index.html")

# Define the route for handling the prediction logic.
@app.route("/predict", methods=["POST"])
def predict():
    """
    This function is called when the user submits the form.
    It takes the input values, runs them through the model, and returns the prediction.
    """
    try:
        # Get the input values from the form and convert them to floating-point numbers.
        gpa = float(request.form['gpa'])
        interview_score = float(request.form['interview_score'])
        
        # Create a NumPy array from the inputs, as this is the format the model expects.
        features = [np.array([gpa, interview_score])]
        
        # Use the loaded model to make a prediction.
        prediction = model.predict(features)
        
        # --- Prepare the Prediction Text ---
        # Based on the model's output (0 or 1), create a user-friendly message.
        # We access the first element of the prediction array with prediction[0]
        if prediction[0] == 1:
            result_text = "Congratulations! The student is likely to be PLACED."
        else:
            result_text = "The student might NOT be placed. Further preparation is recommended."
            
        # Render the HTML page again, but this time with the prediction result.
        return render_template("index.html", prediction_text=result_text)

    except (ValueError, KeyError):
        # Handle cases where input is missing or not a valid number.
        error_text = "Invalid input. Please enter valid numbers for both fields."
        return render_template("index.html", prediction_text=error_text)


# --- Run the Application ---
# This block ensures the app runs only when the script is executed directly.
if __name__ == "__main__":
    # The 'debug=True' argument provides detailed error messages, which is helpful during development.
    app.run(debug=True)

