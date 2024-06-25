import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try : 
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]] 
            
            gs = GridSearchCV(estimator = model, param_grid = para, cv = 5, n_jobs = -1)
            gs.fit(X_train, y_train)
            #model.fit(X_train, y_train)

            model.set_params(**gs.best_params_)     #setting best model parameters
            model.fit(X_train, y_train)

            test_y_pred = model.predict(X_test)

            test_score = r2_score(y_test, test_y_pred)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try :
        with open(file_path, "rb") as file_obj:
            file_obj = dill.load(file_obj)
        return file_obj

    except Exception as e:
        raise CustomException(e, sys)