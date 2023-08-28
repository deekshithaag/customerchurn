import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,StackingClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve, precision_score, recall_score, f1_score, cohen_kappa_score
from src.utils import evaluate_models
from sklearn.ensemble import ExtraTreesClassifier

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                #"Extra Trees Classifier": ExtraTreesClassifier(),
                "XGB Classifier": XGBClassifier(),
                "Bagging Classifier": BaggingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "GradientBoosting Classifier": GradientBoostingClassifier()
            }
            params={
                
                "Random Forest":{
                    'n_estimators':[10,20,30,50],'criterion':['gini','entropy'],'max_depth':[2,3,5,6],
                  'min_samples_split':[2,3,4,5],'min_samples_leaf':[1,3,5],'max_leaf_nodes':[2,3,4]
                },
                "Decision Tree": {
                    'criterion':['gini','entropy'],
                  'max_depth':[2,3,5,6],
                  'min_samples_split':[2,3,4,5],
                  'min_samples_leaf':[1,3,5],
                  'max_leaf_nodes':[2,3,4],
                  'max_features':['auto','sqrt','log2']
                },
                "Logistic Regression":{
                    'penalty':['l1','l2'],'max_iter':[30,50,100,120,150]
                    },
                "XGB Classifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Bagging Classifier":{
                    'base_estimator':[RandomForestClassifier(), LogisticRegression()],
                    'n_estimators':[10, 20, 40],
                    'max_samples':[1,2,3],
                    'max_features':[1,2,3]
                },
                 "AdaBoost Classifier":{
                    'base_estimator':[RandomForestClassifier(),DecisionTreeClassifier()],'n_estimators':[30,50,80,100],
                 'learning_rate':[0.5,1,1.5,2]
                }
                ,
                "GradientBoosting Classifier":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            


        except Exception as e:
            raise CustomException(e,sys)
