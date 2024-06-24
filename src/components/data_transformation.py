import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from logger import logging
from exception import CustomException
from utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


@dataclass
class DataTransformationconfig:
    preprocess_obj_file_path = os.path.join('artifacts', 'preprocess.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for Data Transformation.
        '''
        try : 
            logging.info("Entered data transformation method...")

            numeric_features = ['reading_score', 'writing_score']

            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = "median")),
                    ('scaler' , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = "most_frequent")),
                    ('encoder', OneHotEncoder()),
                ]
            )

            logging.info("Numerical features standard scalling completed...")
            logging.info("Categorical features encodings completed...")

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num', num_pipeline, numeric_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read the train and test data...")

            logging.info("Getting preprocessor object...")

            preprocessor_obj = self.get_data_transformer_object()

            target_column = "math_score"

            target_feature_train_data = train_data[target_column]
            input_features_train_data = train_data.drop(columns = [target_column], axis = 1)

            target_feature_test_data = test_data[target_column]
            input_features_test_data = test_data.drop(columns = [target_column], axis = 1)

            logging.info("transforming input train and test data using preprocessor object...")

            transformed_input_features_train_data  = preprocessor_obj.fit_transform(input_features_train_data)
            transformed_input_features_test_data  = preprocessor_obj.transform(input_features_test_data)              #to not to leak test data info. while training

            #np.c_ ->concatenating array column wise

            train_arr = np.c_[np.array(transformed_input_features_train_data), np.array(target_feature_train_data)]
            test_arr = np.c_[np.array(transformed_input_features_test_data), np.array(target_feature_test_data)]

            logging.info("Saving Preprocessor object")

            save_object(
                file_path = self.data_transformation_config.preprocess_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
