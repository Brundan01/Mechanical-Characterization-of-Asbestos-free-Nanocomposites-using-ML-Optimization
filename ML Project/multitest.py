import pandas as pd
import joblib

# Load the trained models and confusion matrices
rf_classifiers = joblib.load('RF.pkl')
logistic_regs = joblib.load('LOG.pkl')
linear_regs = joblib.load('LR.pkl')

# Load a single data point for testing
test_data_point = {'Tensile': 64, 'Compression': 20}  # Modify with your data
test_data = pd.DataFrame(test_data_point, index=[0])

# Predictions
rf_predictions = {}
logistic_predictions = {}
linear_predictions = {}

for i, target_col in enumerate(['Resin', 'Tamerind', 'Ridge', 'Graphene']):
    # Random Forest Classifier prediction
    rf_prediction = rf_classifiers[i].predict(test_data)[0]
    rf_predictions[target_col] = rf_prediction

    # Logistic Regression prediction
    logistic_prediction = logistic_regs[i].predict(test_data)[0]
    logistic_predictions[target_col] = logistic_prediction

    # Linear Regression prediction
    linear_prediction = linear_regs[i].predict(test_data)[0]
    linear_predictions[target_col] = linear_prediction

# Print predictions
print("Random Forest Classifier Predictions:", rf_predictions)
print("Logistic Regression Predictions:", logistic_predictions)
print("Linear Regression Predictions:", linear_predictions)
