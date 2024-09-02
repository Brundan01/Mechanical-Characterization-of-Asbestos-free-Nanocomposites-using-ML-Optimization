import joblib
import pandas as pd

# Load the trained models
classifier_models = joblib.load('comrs.pkl')

# Load a single data point for testing
test_data_point = {'Tensile strength': 63.42202211, 'Compression strength': 24}  # Modify with your data

# Convert the single data point to a pandas DataFrame
test_data = pd.DataFrame(test_data_point, index=[0])

# Test the Random Forest Classifier
rf_classifier = classifier_models['RandomForestClassifier']
rf_prediction = rf_classifier.predict(test_data)
print("Random Forest Classifier Prediction:", rf_prediction[0])

# Test the Logistic Regression Classifier
logistic_reg = classifier_models['LogisticRegression']
logistic_prediction = logistic_reg.predict(test_data)
print("Logistic Regression Classifier Prediction:", logistic_prediction[0])

# Test the Linear Regression Classifier
linear_reg = classifier_models['LinearRegression']
linear_prediction = linear_reg.predict(test_data)
print("Linear Regression Classifier Prediction:", linear_prediction[0])
