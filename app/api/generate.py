from fastapi import APIRouter
from pydantic import BaseModel

from app.models.generator import Generator

router = APIRouter()

class GeneratePayload(BaseModel):
    filename: str = ''
    prompt: str

@router.post("/")
async def generate_response(payload: GeneratePayload):
    generator = Generator(payload.filename)
    response = generator.generate(payload.prompt)
    return {"response": response}