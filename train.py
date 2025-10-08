import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- Data Loading and Preparation ---
# Define the filename for the dataset
csv_file = r'C:\Users\HP\OneDrive\Desktop\placed\Placement_Data_Full_Class.csv'

# Check if the dataset exists
if not os.path.exists(csv_file):
    print(f"Error: The file '{csv_file}' was not found.")
    print("Please download it from Kaggle and place it in the same directory as this script.")
    # You can add instructions to download the file here if you want
    # For example: print("Download link: https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement")
    exit()

# Load the dataset from the CSV file
df = pd.read_csv(csv_file)

# --- Feature Selection and Preprocessing ---
# We will use 'degree_p' (degree percentage) as the GPA and 'etest_p' as the interview score.
# The target variable is 'status'.
# We select only the columns we need to simplify the model.
df_selected = df[['degree_p', 'etest_p', 'status']].copy()

# Drop any rows that have missing values in our selected columns to ensure data quality.
df_selected.dropna(inplace=True)

# --- Encoding the Target Variable ---
# Machine learning models require numerical input.
# We convert the categorical 'status' column ('Placed'/'Not Placed') into numbers (1/0).
label_encoder = LabelEncoder()
df_selected['status'] = label_encoder.fit_transform(df_selected['status'])

# --- Model Training ---
# Define our features (X) and the target (y).
X = df_selected[['degree_p', 'etest_p']].values
y = df_selected['status'].values

# Split the data into a training set (to teach the model) and a testing set (to evaluate it).
# 80% of the data will be for training, 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We will use Logistic Regression, a simple and effective model for binary classification.
model = LogisticRegression()

# Train the model using our training data.
model.fit(X_train, y_train)

# --- Saving the Model ---
# Now that the model is trained, we save it to a file named 'model.pkl'.
# This allows our web app to use the trained model without retraining it every time.
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# --- Verification ---
# (Optional) Print the model's accuracy on the test data to see how well it performs.
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully and saved as model.pkl")
print(f"Model Accuracy: {accuracy:.2f}")

