import asyncio
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from coordinator import TrainingCoordinator

app = FastAPI(title="Vigil API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

coordinator = None
training_task = None


@app.get("/")
def root():
    return {"status": "Vigil API is running"}


@app.post("/start-training")
async def start_training():
    global coordinator, training_task
    coordinator = TrainingCoordinator(use_chain=True)

    async def run_training():
        for _ in range(coordinator.total_rounds):
            coordinator.run_round()
            await asyncio.sleep(1)

    training_task = asyncio.create_task(run_training())
    return {"message": "Training started", "total_rounds": coordinator.total_rounds}


@app.get("/status")
def get_status():
    if coordinator is None:
        return {
            "current_round": 0,
            "total_rounds": 0,
            "is_training": False,
            "round_history": [],
        }
    return coordinator.get_status()


@app.get("/round/{round_id}")
def get_round(round_id: int):
    if coordinator is None or round_id > len(coordinator.round_history):
        return {"error": "Round not found"}
    return coordinator.round_history[round_id - 1]


class PredictRequest(BaseModel):
    image: list  # 28x28 flat or nested list


@app.post("/predict")
def predict(req: PredictRequest):
    if coordinator is None:
        return {"error": "Model not trained yet"}
    arr = np.array(req.image, dtype=np.float32).reshape(1, 1, 28, 28)
    tensor = torch.from_numpy(arr)
    return coordinator.predict(tensor)
