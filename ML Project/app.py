from flask import Flask, render_template, request, redirect, url_for
from flask_cors import cross_origin
import pandas as pd
import joblib

app = Flask(__name__, template_folder="template")

# Load the trained models
rf_classifiers = joblib.load('RF.pkl')
logistic_regs = joblib.load('LOG.pkl')
linear_regs = joblib.load('LR.pkl')

print("Models Loaded")


@app.route("/")
@cross_origin()
def start():
    return render_template("start.html")


@app.route("/home", methods=['GET', 'POST'])
@cross_origin()
def home():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template("home.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        # Extract data from form
        ets = float(request.form['TS'])
        ecs = float(request.form['CS'])

        # Create a DataFrame from the input data
        test_data_point = {'Tensile': ets, 'Compression': ecs}
        test_data = pd.DataFrame([test_data_point])

        # Prepare lists to store predictions
        rf_predictions = []
        logistic_predictions = []
        linear_predictions = []

        # Define target columns corresponding to model outputs
        target_columns = ['Resin', 'Tamarind Seed Powder', 'Ridge Gourd Powder',
                          'Graphene Powder']  # Replace with actual target column names

        for i in range(len(target_columns)):
            rf_prediction = rf_classifiers[i].predict(test_data)[0]
            rf_predictions.append(abs(rf_prediction))

            logistic_prediction = logistic_regs[i].predict(test_data)[0]
            logistic_predictions.append(abs(logistic_prediction))

            linear_prediction = linear_regs[i].predict(test_data)[0]
            linear_predictions.append(abs(linear_prediction))

        print("Random Forest Classifier Predictions:", rf_predictions)
        print("Logistic Regression Predictions:", logistic_predictions)
        print("Linear Regression Predictions:", linear_predictions)

        return render_template("sunny.html",
                               rf_predictions=rf_predictions,
                               logistic_predictions=logistic_predictions,
                               linear_predictions=linear_predictions,
                               target_columns=target_columns,
                               ts=ets, cs=ecs)

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
