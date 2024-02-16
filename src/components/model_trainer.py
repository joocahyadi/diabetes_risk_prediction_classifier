import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info('')
        except Exception as e:
            raise CustomException(e, sys.exc_info())