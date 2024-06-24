import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from exception import CustomException

def save_object(file_path, obj):
    try : 
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i,j in enumerate(list(models.values())):
            model = j

            model.fit(X_train, y_train)

            test_y_pred = model.predict(X_test)

            test_score = r2_score(y_test, test_y_pred)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    