import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    Initialize some configs
    '''
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
    

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    '''
    This class will map the 'Yes' and 'No' labels, as well as 'Male' and 'Female'
    into 1 and 0 respectively.
    '''
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X.replace({'Yes':1, 'No':0,
                          'Male':1, 'Female':0,
                          'Positive':1, 'Negative':0})

class DataTransformation:
    '''
    This class containes pipelines that will transform the data into the specified format
    '''

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        logging.info('The data transformation process is starting')

        try:
            # Get the numerical and caategorical features
            numerical_features = ['Age']
            categorical_features = ['Gender','Polyuria','Polydipsia','sudden weight loss','weakness',
                                    'Polyphagia','Genital thrush','visual blurring','Itching','Irritability',
                                    'delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','class']

            # Define the preprocessing steps or pipeline for numerical and categorical columns
            numerical_pipeline = 'passthrough'
            categorical_pipeline = Pipeline(steps=[
                ('label_encoder', CustomLabelEncoder())
            ])

            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    

    def initiate_data_transformation(self, train_path, test_path):

        try:

            # Import the train and test data
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info('Successfully imported the train and test data')
            logging.info('Getting the preprocessing object')

            # Get the preprocessing object
            preprocessing_object = self.get_data_transformer_object()

            logging.info('Applying the preprocessor object to the train and test data')

            # Apply the preprocessor object
            df_train_preprocessed = preprocessing_object.fit_transform(df_train)
            df_test_processed = preprocessing_object.transform(df_test)

            logging.info('The numerical and categoircal columns are transformed')

            # Save the preprocessor object
            save_object(file_path=self.data_transformation_config.preprocessor_path,
                        object=preprocessing_object )
            
            logging.info('The preprocessor object has been saved')

            # Return the preprocessed datasets
            return (df_train_preprocessed, df_test_processed, self.data_transformation_config.preprocessor_path)


        except Exception as e:
            raise CustomException(e, sys.exc_info())
