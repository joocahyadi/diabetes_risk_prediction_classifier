import os
import sys

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, train_evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info('Split the features and target on the transformed train and test data')

            # Split the features and target
            X_train, y_train, X_test, y_test = (
                train_data[:,:-1], train_data[:,-1], test_data[:,:-1], test_data[:,-1]
            )

            # Define the ML models
            models = {
                'Logistic Regression': LogisticRegression(),
                'SVM Classifier': SVC(),
                'XGBoost Classifier': XGBClassifier()
            }

            # Define the models' hyperparameters space
            params_grid = {

                # For logistic regression
                'Logistic Regression': {
                    'penalty': ['l1','l2'],
                    'C': [0.01, 0.1, 1.0, 10, 100],
                    'solver': ['liblinear','saga']
                },

                # For SVM Classifier
                'SVM Classifier': {
                    'C': [0.01, 0.1, 1.0, 10, 100],
                    'kernel': ['linear','poly'],
                    'degree': [2,3],
                },

                # For XGBoost Classifier
                'XGBoost Classifier': {
                    'max_depth': [3,6,10],
                    'eta': [0.05, 0.1, 0.15, 0.2]
                }
            }
            
            # Train and get the evaluation of each model
            report, best_model, best_estimator, best_params, train_score = train_evaluate_models(
                X_train, X_test, y_train, y_test, models, param=params_grid, scoring='accuracy'
                )

            # Get the best model score (by accuracy)
            best_model_score = max(report.items(), key=lambda x: x[1])

            logging.info(f'The best model is: {best_model_score[0]} using {best_params} as the set of hyperparameters, with an accuracy of {train_score} in the training set and {best_model_score[1]} in the test set.')

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        object=best_estimator)
            
            logging.info('The training phase has done')

            # Return
            return f'The best model is: {best_model_score[0]} using {best_params} as the set of hyperparameters, with an accuracy of {train_score} in the training set and {best_model_score[1]} in the test set.'

        except Exception as e:
            raise CustomException(e, sys.exc_info())