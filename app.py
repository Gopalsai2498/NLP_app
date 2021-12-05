# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 05:52:46 2021

@author: anil.ms
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib

#cd /d d:/data_science/app/nlp_prediction


#load the model
clf_nb = pickle.load(open('nlp_model.pkl', 'rb'))
cv = pickle.load(open('transform.pickle', 'rb'))

#create flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vec = cv.transform(data).toarray()
        my_prediction = clf_nb.predict(vec)
    return render_template('result.html', prediction= my_prediction)
        
    
#main function will run the whole flask
if __name__ == '__main__':
    app.run(debug=True)


