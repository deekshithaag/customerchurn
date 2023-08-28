import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts',"model.pkl")
            preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")
            print('Before loading')
            model=load_object(file_path=model_path)
            print('model loaded')
            preprocessor=load_object(file_path=preprocessor_path)
            print('After loading')
            print('started')
            data_scaled=preprocessor.transform(features)
            print('Data scaled')
            preds=model.predict(data_scaled)
            print(preds)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 AGE:int,
                 CUS_Month_Income:int,
                 CUS_Gender:str,
                 CUS_Marital_Status:str,
                 YEARS_WITH_US:int,
                 total_debit_amount:int,
                 total_debit_transactions:int,
                 total_credit_amount:int,
                 total_credit_transactions:int,
                 total_transactions:int,
                 TAR_Desc:str,
                 total_amount:int
                 ):
        self.AGE=AGE
        self.CUS_Month_Income=CUS_Month_Income
        self.CUS_Gender=CUS_Gender
        self.CUS_Marital_Status=CUS_Marital_Status
        self.YEARS_WITH_US=YEARS_WITH_US
        self.total_debit_amount=total_debit_amount
        self.total_debit_transactions=total_debit_transactions
        self.total_credit_amount=total_credit_amount
        self.total_credit_transactions=total_credit_transactions
        self.total_transactions=total_transactions
        self.TAR_Desc=TAR_Desc
        self.total_amount=total_amount

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "AGE": [self.AGE],
                "CUS_Month_Income": [self.CUS_Month_Income],
                "CUS_Gender": [self.CUS_Gender],
                "CUS_Marital_Status": [self.CUS_Marital_Status],
                "YEARS_WITH_US": [self.YEARS_WITH_US],
                "total_debit_amount": [self.total_debit_amount],
                "total_debit_transactions": [self.total_debit_transactions],
                "total_credit_amount": [self.total_credit_amount],
                "total_credit_transactions": [self.total_credit_transactions],
                "total_transactions": [self.total_transactions],
                "TAR_Desc": [self.TAR_Desc],
                "total_amount": [self.total_amount],
                

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
