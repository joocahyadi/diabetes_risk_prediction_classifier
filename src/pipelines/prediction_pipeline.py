import sys
import os
import json
import requests
import numpy as np

import pandas as pd
import pickle

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    '''
    This class acts as the pipeline of the whole process. From preprocessing the users' inputs 
    to predicting and returning the output
    '''
    def __init__(self):
        pass

    def predict(self, data):
        try:
            # Define the backend url
            url = "http://127.0.0.1:8000/predict"

            # Get the already fitted preprocessor and trained model
            preprocessor = load_object('artifacts/preprocessor.pkl')

            # Preprocess the input
            preprocessed_inputs = preprocessor.transform(data)

            # Convert the preprocessed input (np array) to json
            preprocessed_inputs_json = json.dumps(preprocessed_inputs.tolist())

            # test = json.loads(preprocessed_inputs_json)

            # Send the input data to the backend and get the prediction result
            response = requests.post(url=url, json={"data": preprocessed_inputs_json})

            # Convert the result from json to python dictionary
            result = json.loads(response.content)

            # Change the ML raw result into Yes or No and return the result
            # 1 -> Yes
            # 0 -> No
            if result["predicted_value"] == 1:
                return "Yes"
            else:
                return "No"
            # return result
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())


class CustomData:
    '''
    This class is responsible to receive and handle users' inputs (from UI), and send them to the backend
    '''

    def __init__(self, age: int, gender: str, polyuria: int, polydipsia: int, sudden_weight_loss: int, 
                 weakness: int, polyphagia: int, genital_thrush: int, visual_blurring: int, itching: int, 
                 irritability: int, delayed_healing: int, partial_paresis: int, muscle_stiffness: int, 
                 alopecia: int, obesity: int):
        self.age = age
        self.gender = gender
        self.polyuria = polyuria
        self.polydipsia = polydipsia
        self.sudden_weight_loss = sudden_weight_loss
        self.weakness = weakness
        self.polyphagia = polyphagia
        self.genital_thrush = genital_thrush
        self.visual_blurring = visual_blurring
        self.itching = itching
        self.irritability = irritability
        self.delayed_healing = delayed_healing
        self.partial_paresis = partial_paresis
        self.muscle_stiffness = muscle_stiffness
        self.alopecia = alopecia
        self.obesity = obesity
    
    def convert_into_dataframe(self):
        '''
        This function will store the inputs in dictionary (json format)
        '''
        try:
            input_dict = {
                'age': [self.age],
                'gender': [self.gender],
                'polyuria': [self.polyuria],
                'polydipsia': [self.polydipsia],
                'sudden_weight_loss': [self.sudden_weight_loss],
                'weakness': [self.weakness],
                'polyphagia': [self.polyphagia],
                'genital_thrush': [self.genital_thrush],
                'visual_blurring': [self.visual_blurring],
                'itching': [self.itching],
                'irritability': [self.irritability],
                'delayed_healing': [self.delayed_healing],
                'partial_paresis': [self.partial_paresis],
                'muscle_stiffness': [self.muscle_stiffness],
                'alopecia': [self.alopecia],
                'obesity': [self.obesity]
            }

            return pd.DataFrame(input_dict)
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())