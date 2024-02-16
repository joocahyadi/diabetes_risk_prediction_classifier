import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException

# Function to save objects
def save_object(file_path, object):
    """
    This function is intended to save objects, like preprocessor and ML models
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