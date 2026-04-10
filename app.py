import sys
sys.stdout.reconfigure(encoding='utf-8')

import gradio as gr
import random
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from models import ClinicalAction, FluidAmount, AntibioticChoice
from tasks import make_env, grade
from environment import SepsisEnvironment


api = FastAPI()
api_env = SepsisEnvironment(task="easy", seed=42)
api_env.reset()


@api.get("/")
def root():
    return {
        "status": "ok",
        "environment": "sepsis-icu-openenv",
        "endpoints": ["/reset", "/step", "/state"]
    }


@api.post("/reset")
def api_reset(task: str = "easy"):
    global api_env
    api_env = SepsisEnvironment(task=task, seed=42)
    state = api_env.reset()
    return JSONResponse(state.model_dump())


@api.get("/state")
def api_state():
    return JSONResponse(api_env.state().model_dump())


@api.post("/step")
def api_step(action: dict):
    act = ClinicalAction(**action)
    result = api_env.step(act)
    return JSONResponse({
        "state": result.state.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    })


env = None
done = False


def reset_environment(task_id):
    global env, done
    env = make_env(task_id, seed=42)
    env.reset()
    done = False
    return "Environment reset!"


def take_action():
    global env, done

    if env is None:
        return "Reset first"

    if done:
        return "Episode finished"

    action = ClinicalAction(
        give_fluids=FluidAmount.SMALL,
        antibiotic=AntibioticChoice.PIP_TAZO,
        start_vasopressors=False,
        stop_vasopressors=False,
        order_labs=False,
        escalate_care=False,
    )

    result = env.step(action)
    done = result.done

    return f"Step reward: {round(result.reward, 3)}"


def run_random_agent(task_id):
    global env, done

    env = make_env(task_id, seed=42)
    env.reset()
    done = False

    while not done:
        action = ClinicalAction(
            give_fluids=random.choice(list(FluidAmount)),
            antibiotic=random.choice(list(AntibioticChoice)),
            start_vasopressors=random.choice([True, False]),
            stop_vasopressors=random.choice([True, False]),
            order_labs=False,
            escalate_care=False,
        )
        result = env.step(action)
        done = result.done

    episode_result = env.get_episode_result()
    score = grade(episode_result)

    return f"Random Agent Score: {score}"


with gr.Blocks() as demo:

    gr.Markdown("# Sepsis ICU OpenEnv")

    task = gr.Dropdown(["easy", "medium", "hard"], value="easy")

    reset_btn = gr.Button("Reset")
    step_btn = gr.Button("Step")
    random_btn = gr.Button("Run Random Agent")

    output = gr.Textbox()

    reset_btn.click(reset_environment, inputs=task, outputs=output)
    step_btn.click(take_action, outputs=output)
    random_btn.click(run_random_agent, inputs=task, outputs=output)


app = gr.mount_gradio_app(api, demo, path="/ui")


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=7860)
