# app/main.py
from dotenv import load_dotenv
from fastapi import FastAPI
from app.api import train, generate


load_dotenv(".env_vars")

app = FastAPI(
    title="Modelo API",
    description="API para treinamento e geração de respostas utilizando modelos de aprendizado de máquina.",
    version="1.0.0",
)

app.include_router(train.router, prefix="/train", tags=["Treinamento"])
app.include_router(generate.router, prefix="/generate", tags=["Geração"])