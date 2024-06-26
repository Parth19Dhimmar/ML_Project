import sys
import os
from exception import CustomException
from logger import logging

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import Modeltrainer

data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation()
train_data, test_data, transformer_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

model_trainer = Modeltrainer()
best_r2_score = model_trainer.initiate_model_trainer(train_data, test_data, transformer_file_path)

print(best_r2_score)