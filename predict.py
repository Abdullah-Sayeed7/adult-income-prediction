import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load('model.bin')

app = FastAPI(title="Adult Income Prediction API")


class CustomerData(BaseModel):
    age: int
    workclass: str
    education_num: int  # Note: we use underscore here for Python
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Adult Income Prediction API! Send a POST request to /predict"}


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([{
        'age': data.age,
        'workclass': data.workclass,
        'education-num': data.education_num,
        'marital-status': data.marital_status,
        'occupation': data.occupation,
        'relationship': data.relationship,
        'race': data.race,
        'sex': data.sex,
        'capital-gain': data.capital_gain,
        'capital-loss': data.capital_loss,
        'hours-per-week': data.hours_per_week,
        'native-country': data.native_country
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    income_label = ">50K" if prediction == 1 else "<=50K"

    return {
        "income_prediction": income_label,
        "probability_over_50k": round(float(probability), 4)
    }
