
import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

#use pickle to lad in the pre-trained model
app = Flask(__name__)

with open(f'model/stroke_logreg_final88.pkl','rb') as f:
    model = pickle.load(f)

@app.route('/')

def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])

def predict():    
    age = flask.request.form['age']
    avg_glucose_level = flask.request.form['avg_glucose_level']
    gender = flask.request.form['gender']
    hypertension = flask.request.form['hypertension']
    heart_disease = flask.request.form['heart_disease']
    ever_married = flask.request.form['ever_married']
    work_type = flask.request.form['work_type']
    residence_type = flask.request.form['residence_type']
    smoking_status = flask.request.form['smoking_status']
    
    input_variables = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, smoking_status]],
                                    columns=["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "residence_type", "avg_glucose_level", "smoking_status"])
    prediction = model.predict(input_variables)
    print(prediction)
    return flask.render_template('result.html', prediction_text = prediction)       
    
@app.route("/analysis")
def analysis():
    return render_template('analysis.html')

@app.route("/data")
def data():
    return render_template('data.html')

if __name__ == '__main__':
    app.run()
