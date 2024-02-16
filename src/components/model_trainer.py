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
            
            # Train and get the evaluation of each model
            report, best_model = train_evaluate_models(X_train, X_test, y_train, y_test, models)

            # Get the best model score (by accuracy)
            best_model_score = max(report.items(), key=lambda x: x[1])

            logging.info(f'The best model is: {best_model_score[0]}, with an accuracy of {best_model_score[1]}')

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        object=best_model)
            
            logging.info('The training phase has done')

            # Return
            return f'The best model is: {best_model_score[0]}, with an accuracy of {best_model_score[1]}'

        except Exception as e:
            raise CustomException(e, sys.exc_info())