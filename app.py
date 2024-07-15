from flask import Flask, render_template, request,  redirect, url_for
import numpy as np
from joblib import load
import requests
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer


app = Flask(__name__)


# loading the saved models
diabetes_model = load('diabetes/diabetes_model.joblib')
heart_disease_model = load('heart/heart_disease_model.joblib')

# Set up Hugging Face GPT-2 model
gpt2_model_name = "gpt2"  # You can choose a different GPT-2 variant if needed
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)


def generate_lifestyle_recommendations(diagnosis, risk_level):
    prompt = f"As per the diagnosis, the patient has a {risk_level}% risk of {diagnosis}. Based on this, I recommend the following lifestyle changes: "

    # Tokenize the prompt
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the GPT-2 model
    output = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode the generated text
    lifestyle_recommendations = gpt2_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    return lifestyle_recommendations







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

    # Generate lifestyle recommendations based on the diagnosis and risk level
    lifestyle_recommendations = generate_lifestyle_recommendations("diabetes", diabetes_level)

    result = f"The probability of diabetes is {diabetes_probability}\n"
    result += f"You are at a {diabetes_level}% risk of diabetes.\n"
    result += f"Lifestyle Recommendations: {lifestyle_recommendations}"

    return result

# the heart_disease_prediction function
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

    # Generate lifestyle recommendations based on the diagnosis and risk level
    lifestyle_recommendations = generate_lifestyle_recommendations("heart disease", heart_disease_level)

    result = f"The probability of heart disease is {heart_disease_probability}\n"
    result += f"You are at a {heart_disease_level}% risk of heart disease.\n"
    result += f"Lifestyle Recommendations: {lifestyle_recommendations}"

    return result

@app.route('/')
def home():
    return render_template('/index.html')

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

        # Redirect to diab_results route with the diagnosis as a query parameter
        return redirect(url_for('diab_results', diagnosis=diab_diagnosis))
    else:
        return render_template('diabetes.html')

@app.route('/diab_results')
def diab_results():
    # Retrieve the diagnosis query parameter from the URL
    diab_diagnosis = request.args.get('diagnosis')

    return render_template('diab_results.html', diagnosis=diab_diagnosis)


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

        # Redirect to heart_results route with the diagnosis as a query parameter
        return redirect(url_for('heart_results', diagnosis=heart_diagnosis))
    else:
        return render_template('heart_disease.html')

@app.route('/heart_results')
def heart_results():
    # Retrieve the diagnosis query parameter from the URL
    heart_diagnosis = request.args.get('diagnosis')

    return render_template('heart_results.html', diagnosis=heart_diagnosis)

if __name__ == '__main__':
    app.run(debug=True)
