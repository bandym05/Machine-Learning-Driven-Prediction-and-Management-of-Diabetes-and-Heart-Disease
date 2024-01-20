from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)

# loading the saved models
diabetes_model = load('diabetes/diabetes_model.joblib')
heart_disease_model = load('heart/heart_disease_model.joblib')

# Modify the diabetes_prediction function
def diabetes_prediction(input_data):
    # Convert the data point to a NumPy array
    input_data_np = np.asarray(input_data)

    # Reshape the array to fit the model's input format
    input_data_reshaped = input_data_np.reshape(1, -1)

    # Standardize the input data
    scaler = load('diabetes/scaler.joblib')
    std_data = scaler.transform(input_data_reshaped)

    # Predict the probability of diabetes
    prediction = diabetes_model.predict_proba(std_data)

    # Calculate the risk percentage
    diabetes_probability = prediction[0, 1] * 100

    # Add the new part
    diabetes_threshold = 0.5  # Adjust this threshold based on your model and requirements
    diabetes_level = "High" if diabetes_probability > diabetes_threshold else "Low"

    result = f"The probability of diabetes is {diabetes_probability}\n"
    result += f"The patient has a {diabetes_level}% risk of diabetes."

    return result

# Change the heart_disease_prediction function
def heart_disease_prediction(input_data):
    # Convert the data point to a NumPy array
    input_data_np = np.asarray(input_data)

    # Reshape the array to fit the model's input format
    input_data_reshaped = input_data_np.reshape(1, -1)

    # Standardize the input data
    scaler = load('heart/scaler.joblib')
    std_data = scaler.transform(input_data_reshaped)

    # Predict the probability of heart disease
    prediction = heart_disease_model.predict_proba(std_data)

    # Calculate the risk percentage
    heart_disease_probability = prediction[0, 1] * 100

    # Add the new part
    heart_disease_threshold = 0.5  # Adjust this threshold based on your model and requirements
    heart_disease_level = "High" if heart_disease_probability > heart_disease_threshold else "Low"

    result = f"The probability of heart disease is {heart_disease_probability}\n"
    result += f"The patient has a {heart_disease_level}% risk of heart disease."

    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = 1 if request.form['Gender'] == '1' else 0
        Polyuria = 1 if request.form['Polyuria'] == '1' else 0
        Polydipsia = 1 if request.form['Polydipsia'] == '1' else 0
        Weightloss = 1 if request.form['Weightloss'] == '1' else 0
        weakness = 1 if request.form['weakness'] == '1' else 0
        Polyphagia = 1 if request.form['Polyphagia'] == '1' else 0
        Genitalthrush = 1 if request.form['Genitalthrush'] == '1' else 0
        visualblurring = 1 if request.form['visualblurring'] == '1' else 0
        Itching = 1 if request.form['Itching'] == '1' else 0
        Irritability = 1 if request.form['Irritability'] == '1' else 0
        delayedhealing = 1 if request.form['delayedhealing'] == '1' else 0
        partialparesis = 1 if request.form['partialparesis'] == '1' else 0
        musclestiffness = 1 if request.form['musclestiffness'] == '1' else 0
        Alopecia = 1 if request.form['Alopecia'] == '1' else 0
        Obesity = 1 if request.form['Obesity'] == '1' else 0

        diab_diagnosis = diabetes_prediction([Age, Gender, Polyuria, Polydipsia, Weightloss, weakness, Polyphagia,
                                              Genitalthrush, visualblurring, Itching, Irritability, delayedhealing,
                                              partialparesis, musclestiffness, Alopecia, Obesity])

        return render_template('diabetes.html', diagnosis=diab_diagnosis)
    else:
        return render_template('diabetes.html')


@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        heart_diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

        return render_template('heart_disease.html', diagnosis=heart_diagnosis)
    else:
        return render_template('heart_disease.html')

if __name__ == '__main__':
    app.run(debug=True)
