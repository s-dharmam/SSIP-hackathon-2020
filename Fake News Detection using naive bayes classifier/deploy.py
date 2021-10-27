from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import model
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app)
@app.route("/")
def hello():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        my_prediction=model.predict.get_result(message)[0]
    return render_template('result.html', prediction = my_prediction)

    
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)
