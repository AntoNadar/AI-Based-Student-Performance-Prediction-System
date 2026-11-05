import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Loading the Dataset
data = pd.read_csv("student_performance_encoded.csv")

# Separate input features
X = data[['Attendance','Hours_Studied','Sleep_Hours','Previous_Scores','Tutoring_Sessions','Motivation_Level','Parental_Involvement','Access_to_Resources']]
# Target variable
y = data['Exam_Score']

# Splitting the dataset into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model,'ai_student_model.pkl')
print("Model Trained and Saved successfully!")


features = X.columns.tolist()
joblib.dump(features, "student_features.pkl")