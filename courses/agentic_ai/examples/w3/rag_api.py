#!/usr/bin/env python3
from fastapi import FastAPI
import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions


genai_api_key = "AIzaSyCq8PAsEMxLQbOTBAV4KEsjv9KGxmQl3Xs"
eb_model = "all-MiniLM-L6-v2"
genai_model = "gemini-3-pro-preview"
model = SentenceTransformer(eb_model)
chroma_client = chromadb.PersistentClient(path="chormadb3")
collection = chroma_client.get_or_create_collection(name="document_embeddings")

model = SentenceTransformer(eb_model)
def gen_embedding(text):    
    embedding = model.encode(text)
    return embedding


gemini_client = genai.Client(
    api_key=genai_api_key,
    http_options=types.HttpOptions(api_version='v1beta'),
    )

def call_genai(question, relevent_chunks):
    context = "\n".join(relevent_chunks)
    prompt = f"""
    You are a helpful financial assistant. Answer the company financial question based on the context provided.
    Context: {context}
    Question: {question}    
    Answer:
    """
    response = gemini_client.models.generate_content(
        model=genai_model,
        contents=prompt
    )
    return response



app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "echo"}

class Question(BaseModel):
    q: str

@app.post("/question")
def answer(q: Question):
    query_embedding = gen_embedding(q.q)

    results = collection.query(
        query_embeddings=[query_embedding],
        include=["documents", "metadatas"],
        n_results=4
    )
    # find relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    answer = call_genai(q, relevant_chunks)
    return {"question": q.q, "answer": answer.text}

# source .venv/bin/activate
# Run with: uvicorn rag_api:app --reload
