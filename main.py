import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Dataset load
data = pd.read_csv("student_data.csv")

# Inputs
X = data[["study_hours", "attendance", "sleep_hours", "previous_marks"]]

# Output
y = data["final_marks"]

# Split training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Test prediction
predictions = model.predict(X_test)

# Error
error = mean_absolute_error(y_test, predictions)

print("Average Error:", error)

# User Input
study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance %: "))
sleep_hours = float(input("Enter Sleep Hours: "))
previous_marks = float(input("Enter Previous Marks: "))

# Final prediction
result = model.predict([[study_hours, attendance, sleep_hours, previous_marks]])

print("\nPredicted Marks:", round(result[0], 2))