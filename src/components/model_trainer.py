import os
import sys
from dataclasses import dataclass
from exception import CustomException
from logger import logging
from utils import evaluate_models, save_object

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
    )
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splittting the train and test data...")

            X_train, X_test, y_train, y_test = (
                train_array[ : , :-1],
                test_array[ : , :-1 ],
                train_array[ : , -1],
                test_array[ : , -1]
            )

            models = {'Linear Regression' : LinearRegression(),
                      'Logistic Regression' : LogisticRegression(), 
                      'K Nearest Neighbor' : KNeighborsRegressor(),
                      'Decision Tree Regressor'  : DecisionTreeRegressor(),
                      'SVR' : SVR(),
                      'Random Forest Regressor' : RandomForestRegressor(),
                      'Gradient Boosting': GradientBoostingRegressor(),
                      'Ada Boost' : AdaBoostRegressor(),
            }
            
            params = {

                "Linear Regression" : {},

                "Logistic Regression" : {
                    'penalty' : ['l1', 'l2']
                },

                "K Nearest Neighbor" : {
                    'n_neighbors' : [5, 10, 20, 50, 100],
                    #'weights' : ['uniform', 'distance'],
                    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                },

                "Decision Tree Regressor" : {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter' : ['best', 'random'],
                    'max_depth' : [None, 5, 10, 20],
                    'max_features' : [None, 'sqrt', 'log2'],
                },

                "SVR" : {
                    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid',],
                    'gamma' : ['scale', 'auto'],
                },

                "Random Forest Regressor" : {
                    'n_estimators' : [10, 20, 50, 100],
                    'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'max_depth' : [2, 5, 10, 15, 25],
                },

                "Gradient Boosting" : {
                    'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators' : [10, 20, 50, 100],
                    'criterion' : ['squared_error', 'friedman_mse',],
                    'max_depth' : [2, 5, 10, 15, 25],
                    'max_features' : [None, 'sqrt', 'log2'],
                },

                "Ada Boost" : {
                    'n_estimators' : [10, 20, 50, 100],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'loss' : ['linear', 'square', 'exponential'],
                }
            }

            logging.info("Model is being trained on data and evaluated...")

            model_report : dict  = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)

            max_r2_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(max_r2_score)]

            best_model = models[best_model_name] #to get current state of respective model

            if max_r2_score < 0.6:
                raise CustomException("No best model found!")
            
            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            logging.info(f"best model is {best_model} and its r2score is {max_r2_score}...")

            return max_r2_score

        except Exception as e:
            raise CustomException(e, sys)
