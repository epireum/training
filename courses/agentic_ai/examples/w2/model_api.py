#!/usr/bin/env python3
from fastapi import FastAPI
import pickle

with open("salarymodel.pkl", "rb") as f:
    loaded_model = pickle.load(f)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "echo"}

@app.get("/experience/{experience}")
def read_item(experience: int):
    salary = loaded_model.predict([[experience]])
    salary_value = salary[0]
    return {"experience": experience, "salary": salary_value}

# Run with: uvicorn model_api:app --reload
