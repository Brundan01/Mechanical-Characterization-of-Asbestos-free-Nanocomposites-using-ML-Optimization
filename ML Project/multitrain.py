import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import joblib

# Load data from CSV
data = pd.read_csv('findataset1.csv')  # Replace 'your_data.csv' with the path to your dataset

# Preprocessing
X = data[['Tensile', 'Compression']]
y = data[['Resin', 'Tamerind', 'Ridge', 'Graphene']]  # Adjust the column names as per your dataset

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a classifier for each target column
rf_classifiers = []
logistic_regs = []
linear_regs = []

for target_col in y.columns:
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train[target_col])
    rf_classifiers.append(rf_classifier)

    # Logistic Regression
    logistic_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logistic_reg.fit(X_train, y_train[target_col])
    logistic_regs.append(logistic_reg)

    # Linear Regression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train[target_col])
    linear_regs.append(linear_reg)

# Save models
joblib.dump(rf_classifiers, 'RF.pkl')
joblib.dump(logistic_regs, 'LOG.pkl')
joblib.dump(linear_regs, 'LR.pkl')

# Check if models are correctly saved
print("Models saved successfully!")
