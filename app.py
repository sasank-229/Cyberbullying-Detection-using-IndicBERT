from flask import Flask, request, render_template,url_for
from tqdm import tqdm
import numpy as np
# import nbformat
# from nbconvert import PythonExporter
# import os
import torch
from transformers import AutoModel,AutoTokenizer
import pickle
from xgboost import XGBClassifier

app = Flask(__name__)


from tempCodeRunnerFile import match
from preprocessing import model_extract
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' ,methods=['POST','GET'])
def predict():
    input_string=request.form['text']
    print('text: ',input_string)
    with open('static/ipynbFiles/classifier_10epochs_updated.pkl','rb') as file:
        clf=pickle.load(file)
    with open('static/ipynbFiles/preprocess.pkl', 'rb') as f:
        preprocess_function = pickle.load(f)

    if any(c in input_string for c in match):
        prediction = [0]
    else:
        ans=preprocess_function(input_string)
        print('torch.tensor variable: ',ans)
        prediction = clf.predict(ans)

    print('prediction=',prediction)
    if prediction==[0]:
        return render_template('index.html', pred='Cyberbullying Text', question='వాక్యం -   '+input_string)
    else:
        return render_template('index.html', pred='Non-Cyberbullying Text', question='వాక్యం -   '+input_string)
    