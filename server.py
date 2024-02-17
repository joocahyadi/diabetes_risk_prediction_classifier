# Import Libraries
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle

from src.utils import load_object


# Prediction
# Define the model
def model_prediction(data):

    # Load the model
    model = load_object('artifacts/model.pkl')

    # Encapsulate the value of data in a list, so it can go into model
    # Accepted value example that model can receive: [[42,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
    # new_data = [data['data']]
    new_data = data["data"].strip("[]").split(",")
    new_data = [int(num) for num in new_data]
    new_data = [new_data]

    # Return the prediction result
    return model.predict(new_data)
    # return new_data



# FastAPI backend prediction
app = FastAPI()

@app.post('/predict')
def predict(data: dict):

    # Prediction
    try: 
        # Get the prediction
        response_data = {
            'predicted_value': int(model_prediction(data)[0])
            # 'resultt': model_prediction(data)
        }

        # Return the json-style response
        return JSONResponse(content=response_data)
    
    except Exception as e:
        return JSONResponse(content={'error':str(e)})