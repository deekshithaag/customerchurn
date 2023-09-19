from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            AGE=request.form.get('AGE'),
            CUS_Month_Income=request.form.get('CUS_Month_Income'),
            CUS_Gender=request.form.get('CUS_Gender'),
            CUS_Marital_Status=request.form.get('CUS_Marital_Status'),
            YEARS_WITH_US=request.form.get('YEARS_WITH_US'),
            total_debit_amount=request.form.get('total_debit_amount'),
            total_debit_transactions=request.form.get('total_debit_transactions'),
            total_credit_amount=request.form.get('total_credit_amount'),
            total_credit_transactions=request.form.get('total_credit_transactions'),
            total_transactions=request.form.get('total_transactions'),
            TAR_Desc=request.form.get('TAR_Desc'),
            total_amount=request.form.get('total_amount')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        if results[0]==0:
            res='ACTIVE'
        else:
            res='CHURN'
        print("after Prediction")
        return render_template('home.html',results=res)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)        




