# Machine Learning-Driven Prediction and Management of Diabetes and Heart Disease

Welcome to the repository for the research project "Machine Learning-Driven Prediction and Management of Diabetes and Heart Disease" by Bandile Malaza, submitted for the Bachelor of Science in Information Technology at the University of Eswatini.

# Authors

- **Bandile Malaza**
- **Mr T.V. Magagula (Supervisor)**
- **Dr. Metfula (Secondary Supervisor)**

## Project Overview
#### This project focuses on leveraging machine learning algorithms to predict and manage diabetes and heart disease. The main objective is to develop a web application that uses predictive models to identify high-risk individuals and provide personalized recommendations for preventive care. The project aims to improve healthcare accessibility and empower healthcare providers to take a proactive approach to managing non-communicable diseases.

#### The project includes the development of a user-friendly web application that allows users to input relevant health data for predicting the risk of diabetes and heart disease. The application provides personalized risk percentages, clear risk category labels, and lifestyle modification recommendations based on the predictions.

## Key Features
- **Model Training and Evaluation:** The project involves training machine learning models for diabetes and heart disease prediction using datasets obtained from reputable sources like Kaggle and the UCI Machine Learning Repository. The models are evaluated based on metrics such as accuracy, precision, recall, F1 score, and AUC.
- **Model Deployment:** After training the models using Google Colab, both the diabetes and heart disease models, along with the scalers, were saved to be used in the Flask web application for real-time predictions.
- **Web Application Development:** The project includes the development of a user-friendly web application that allows users to input relevant health data for predicting the risk of diabetes and heart disease. The application provides personalized risk percentages, clear risk category labels, and lifestyle modification recommendations based on the predictions.
- **Ethical Considerations:** The project emphasizes ethical considerations, with disclaimers on system usage and ethical guidelines prominently displayed on the home page for user awareness.

## Tools and Technologies Used
- **Software:** Google Collaboratory, Scikit-learn, Github, Chrome Browser
- **Programming Languages and Frameworks:** Python, Flask, GPT2, Hugging Face Transformers library

## Training of Models and Evaluation
The project involved training machine learning models for heart disease and diabetes prediction. The heart disease prediction model utilized Support Vector Machine (SVM), Logistic Regression, and Decision Tree Classifier algorithms, while the diabetes prediction model showcased the Random Forest Classifier as a standout performer, achieving perfect precision, recall, and F1-score. These models were saved and integrated into the Flask web application for real-time predictions.

## Heart Disease Prediction Model
**Algorithms Used:** Support Vector Machine (SVM), Logistic Regression, Decision Tree Classifier

### Performance Metrics:
- **SVM:** Accuracy of 82%, Precision of 77% for class 1
- **Logistic Regression:** Accuracy of 80%, Precision of 77% for class 1
- **Decision Tree Classifier:** Perfect precision, recall, and F1-score for both classes; 100% accuracy; AUC of 100

## Diabetes Prediction Model
**Algorithms Used:** SVM, Logistic Regression, Decision Tree Classifier, Random Forest Classifier

### Performance Metrics:
- **SVM:** Accuracy of 93% with balanced precision and recall
- **Logistic Regression:** Accuracy of 93% with balanced precision and recall
- **Decision Tree Classifier:** Accuracy of 97% with perfect precision, recall, and F1-score
- **Random Forest Classifier:** Perfect precision, recall, and F1-score for both classes; 100% accuracy; AUC of 100

## Model Comparison and Analysis
A detailed comparative analysis was conducted to identify the best-performing algorithm for predicting heart disease and diabetes. The Decision Tree Classifier and Random Forest Classifier showed remarkable performance, indicating their potential suitability for predicting these diseases.

## Deployment and User Testing
- **Deployment:** The system is deployed using Render and GitHub integration for automatic updates.
- **Accessibility:** Users can access the web application from any standard web browser.
- **User Testing:** Conducted to ensure system usability and immediate response to user inputs.
- **Ethical Considerations:** Disclaimers and ethical considerations are prominently displayed on the home page for user awareness.
  
## User Interface
**Flask Framework:** The web application was developed using the Flask framework, known for its simplicity and efficiency in transforming data scripts into shareable web apps. Flask allowed for rapid prototyping and easy integration with machine learning models.

**Home page**
![unnamed](https://github.com/bandym05/Machine-Learning-based-Diabetes-and-Heart-Disease-Prediction-with-Personalisation-Features/assets/58115126/a64b4ecb-0e17-4179-aa84-218a40d6a138)

![unnamed (1)](https://github.com/bandym05/Machine-Learning-based-Diabetes-and-Heart-Disease-Prediction-with-Personalisation-Features/assets/58115126/a9c20288-442f-4809-a9f4-7e1df2197aa5)

![unnamed (2)](https://github.com/bandym05/Machine-Learning-based-Diabetes-and-Heart-Disease-Prediction-with-Personalisation-Features/assets/58115126/6b9f2dfa-bd7f-4a2d-a73f-af5b6261ece8)

![unnamed (3)](https://github.com/bandym05/Machine-Learning-based-Diabetes-and-Heart-Disease-Prediction-with-Personalisation-Features/assets/58115126/66febb88-308b-4d7e-be7c-29340d4d2add)

## Conclusion
This project aims to empower individuals in managing non-communicable diseases through machine learning-driven predictions. The web application provides a user-friendly interface for predicting diabetes and heart disease risks, contributing to healthcare services in Eswatini and other low- and middle-income settings.
