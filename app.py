from fastapi import FastAPI
from fastapi.responses import JSONResponse

from environment import SepsisEnvironment
from models import ClinicalAction

app = FastAPI()

env = SepsisEnvironment(task="easy", seed=42)
env.reset()


@app.get("/")
def root():
    return {"status": "ok", "environment": "sepsis-icu-openenv"}


@app.post("/reset")
def reset(task: str = "easy"):
    global env
    env = SepsisEnvironment(task=task, seed=42)
    state = env.reset()
    return JSONResponse(state.model_dump())


@app.get("/state")
def state():
    return JSONResponse(env.state().model_dump())


@app.post("/step")
def step(action: dict):
    act = ClinicalAction(**action)
    result = env.step(act)
    return JSONResponse({
        "state": result.state.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    })
