#!/usr/bin/env python3
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "echo"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

@app.post("/users")
def create_user(name: str, age: int):
    return {"name": name, "age": age, "status": "created"}

# Run with: uvicorn fastapi_example:app --reload
