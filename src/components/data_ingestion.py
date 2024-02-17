import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('The data ingestion process is starting')

        try:
            # Read the dataset from csv file format
            df = pd.read_csv('notebook\data\diabetes_risk_prediction_dataset.csv')

            # Lowercase all columns' name
            df.columns = df.columns.str.lower()
            
            # Rename some columns that have name with space
            df = df.rename(columns={'sudden weight loss':'sudden_weight_loss',
                        'visual blurring':'visual_blurring',
                        'genital thrush':'genital_thrush',
                        'delayed healing':'delayed_healing',
                        'partial paresis':'partial_paresis',
                        'muscle stiffness':'muscle_stiffness'
                        })
            
            logging.info('Successfully imported the dataset')

            # Create the folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save df as the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data into train and test
            logging.info('The data splitting process into train and test data is starting')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test data
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('The data ingestion process has completed')

            # Return the paths
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())
        

if __name__ == '__main__':
    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()

    transformation_obejct = DataTransformation()
    train_data_processed, test_data_processed, preprocessor_path = transformation_obejct.initiate_data_transformation(train_path, test_path)

    trainer_object = ModelTrainer()
    print(trainer_object.initiate_model_trainer(train_data_processed, test_data_processed))