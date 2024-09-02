import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix
# Load data from CSV
data = pd.read_csv('findatasetrs.csv')

# Preprocessing
X = data.drop('Resin', axis=1)
y = data['Resin']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)

# Logistic Regression
logistic_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logistic_reg.fit(X_train, y_train)
logistic_y_pred = logistic_reg.predict(X_test)

# Linear Regression (using Logistic Regression for demonstration)
linear_reg = LogisticRegression()
linear_reg.fit(X_train, y_train)
linear_y_pred = linear_reg.predict(X_test)

# Generate Classification Reports
rf_classification_report = classification_report(y_test, rf_y_pred)
logistic_classification_report = classification_report(y_test, logistic_y_pred)
linear_classification_report = classification_report(y_test, linear_y_pred)

print("Classification Report for Random Forest:")
print(rf_classification_report)
print("Classification Report for Logistic Regression:")
print(logistic_classification_report)
print("Classification Report for Linear Regression:")
print(linear_classification_report)

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, rf_y_pred, labels=np.unique(y_test))

# Save models and predictions
joblib.dump(rf_classifier, 'rfrs.pkl')
joblib.dump(logistic_reg, 'logrs.pkl')
joblib.dump(linear_reg, 'lrrs.pkl')

with open('rfrs.pkl', 'wb') as f:
    pickle.dump(rf_y_pred, f)

with open('logrs.pkl', 'wb') as f:
    pickle.dump(logistic_y_pred, f)

with open('lrrs.pkl', 'wb') as f:
    pickle.dump(linear_y_pred, f)
