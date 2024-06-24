import os
import sys
from dataclasses import dataclass
from exception import CustomException
from logger import logging

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
from sklearn.metrics import r2_score
models = {'Linear Regression' : LinearRegression(),
                      'Logistic Regression' : LogisticRegression(), 
                      'K Nearest Neighbor' : KNeighborsRegressor(),
                      'Decision Tree Regressor'  : DecisionTreeRegressor(),
                      'SVR' : SVR(),
                      'Random Forest Regressor' : RandomForestRegressor(),
                      'Gradient Boosting': GradientBoostingRegressor(),
                      'Ada Boost' : AdaBoostRegressor(),
            }

report = {}
for i,j in enumerate(list(models.values())):
    
    print(type(list(models.keys())))
    report[list(models.keys())[i]] = i

print(report)

best_model = list(report.keys())[list(report.values()).index(max(report.values()))]

print(best_model)