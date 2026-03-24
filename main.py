import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

path = os.path.join(os.getcwd(), "model", "encoder.pkl") # DONE: enter the path for the saved encoder 
encoder = load_model(path)

path = os.path.join(os.getcwd(), "model", "model.pkl") # DONE: enter the path for the saved model 
model = load_model(path)

# DONE: create a RESTful API using FastAPI
app = FastAPI()

# DONE: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return{"greeting": "Hello World!"}


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        X=data, # use data as data input
        categorical_features=cat_features,
        encoder=encoder,
        training=False # use training = False
        # do not need to pass lb as input
    )
    # your code here to predict the result using data_processed
    _inference = inference(
        model=model,
        X=data_processed
    )
    return {"result": apply_label(_inference)}
