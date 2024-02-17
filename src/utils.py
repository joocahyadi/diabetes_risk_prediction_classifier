import os
import sys
import pandas as pd
import numpy as np
import dill

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# Function to save objects
def save_object(file_path, object):
    """
    This function is intended to save objects, like preprocessor and ML models.
    """

    try:
        # Get the directory of the file
        dir_path = os.path.dirname(file_path)

        # Create a directory if it hasn't been created
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(object, f)

    except Exception as e:
        raise CustomException(e, sys.exc_info())


# Function to train and evaluate ML models
def train_evaluate_models(X_train, X_test, y_train, y_test, list_models, param, scoring):
    """
    This function is intended to train machine learning models and evaluate the scores corresponding
    to each model.
    """

    try:
        # Initialize empty dictionary to contain all scores from each model
        report = {}
        best_model = None
        best_accuracy = 0

        for i in range(len(list_models)):

            # Get the i-th model
            model = list(list_models.values())[i]

            # Get the corresponding setting of hyperparameters
            params = param[list(list_models.keys())[i]]

            # Fit and train the model using GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring=scoring, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get the best estimator and best set of hyperparameters
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Fit the model
            # model.fit(X_train, y_train)
            
            # Model's prediction
            y_train_pred = best_estimator.predict(X_train)
            y_test_pred = best_estimator.predict(X_test)
            
            # Get the accuracy score and ROC-AUC score
            model_train_score_acc = accuracy_score(y_train, y_train_pred)
            model_test_score_acc = accuracy_score(y_test, y_test_pred)
            # model_train_score_auc = roc_auc_score(y_test, y_pred)

            # Save the scores
            report[list(list_models.keys())[i]] = round(model_test_score_acc, 3)

            if model_test_score_acc > best_accuracy:
                best_accuracy = model_test_score_acc
                best_model = model

        # Return the report
        return report, best_model, best_params, model_train_score_acc

    except Exception as e:
        raise CustomException(e, sys.exc_info())
