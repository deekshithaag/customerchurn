import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import Data_Transformation_Config

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")  

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_excel('notebook/data/CHURNDATA (1).xlsx') 
            logging.info('Read the dataset as a dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train test split initiated')
            df.columns=['CIF','CUS_DOB','AGE','CUS_Month_Income','CUS_Gender','CUS_Marital_Status','CUS_Customer_Since','YEARS_WITH_US','debit_trans_S1','debit_trans_S2','debit_trans_S3','debit_amount_S1','debit_amount_S2','debit_amount_S3','credit_trans_S1','credit_trans_S2','credit_trans_S3','credit_amount_S1','credit_amount_S2','credit_amount_S3','total_debit_amount','total_debit_transactions','total_credit_amount','total_credit_transactions','total_transactions','CUS_Target','TAR_Desc','Status']
            df.drop(['CIF','CUS_DOB','CUS_Customer_Since'],axis=1,inplace=True)
            df.drop(['debit_trans_S1','debit_trans_S2','debit_trans_S3','debit_amount_S1','debit_amount_S2','debit_amount_S3','credit_trans_S1','credit_trans_S2','credit_trans_S3','credit_amount_S1','credit_amount_S2','credit_amount_S3'],axis=1,inplace=True)
            df['CUS_Target']=df['CUS_Target'].astype(object)
            df['total_amount']=df['total_credit_amount']-df['total_debit_amount']
            df['Status']=df['Status'].map({'ACTIVE':0,'CHURN':1})
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

         

