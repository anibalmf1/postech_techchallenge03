from fastapi import APIRouter
from pydantic import BaseModel

from app.models.trainer import train_fine_tune, generate_dataset, verify_dataset, clean_file

router = APIRouter()

class FilenamePayload(BaseModel):
    filename: str

@router.post("/clean")
async def clean(payload: FilenamePayload):
    try:
        new_file_name = clean_file(payload.filename)

        return {"message": "Dataset gerado!", "filename": new_file_name}
    except Exception as e:
        return {"error": str(e)}

@router.post("/dataset")
async def dataset(payload: FilenamePayload):
    try:
        generate_dataset(payload.filename)

        return {"message": "Dataset gerado!"}
    except Exception as e:
        return {"error": str(e)}

@router.post("/dataset/verify")
async def verify(start: int, end: int, payload: FilenamePayload):
    try:
        return verify_dataset(payload.filename, start, end)
    except Exception as e:
        return {"error": str(e)}


class TrainModelPayload(BaseModel):
    filename: str
    sample: int = 0

@router.post("/model")
async def train_model(payload: TrainModelPayload):
    try:
        train_fine_tune(payload.filename, payload.sample)

        return {"message": "Treinamento concluído!"}
    except Exception as e:
        return {"error": str(e)}
