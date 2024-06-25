import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from sklearn.preprocessing import StandardScaler
from pipeline.predict_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET' : 
        return render_template('home.html')
    else:
        custom_data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score'),
        )


    features_df = custom_data.get_data_as_dataframe()

    prediction_pipeline = PredictionPipeline()
    prediction = int(np.round(prediction_pipeline.prediction(features = features_df)[0]))
    
    return render_template('home.html', results = prediction)


if __name__ == "__main__":
    app.run(debug = True)