#FORCE SUBMISSION UPDATE 
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import random
import json
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
from models import ClinicalAction, FluidAmount, AntibioticChoice
from tasks import make_env, grade, TASKS
from environment import SepsisEnvironment

# ─────────────────────────────────────────────
#  FASTAPI REST API
# ─────────────────────────────────────────────

app = FastAPI()
api_env = SepsisEnvironment(task="easy", seed=42)
api_env.reset()

@app.get("/")
def root():
    return {"status": "ok", "environment": "sepsis-icu-openenv"}

@app.post("/reset")
def api_reset(task: str = "easy"):
    global api_env
    api_env = SepsisEnvironment(task=task, seed=42)
    state = api_env.reset()
    return JSONResponse(state.model_dump())

@app.get("/state")
def api_state():
    return JSONResponse(api_env.state().model_dump())

@app.post("/step")
def api_step(action: dict):
    act = ClinicalAction(**action)
    result = api_env.step(act)
    return JSONResponse({
        "state": result.state.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    })

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────

env = None
current_task = "easy"
history = []
done = False
total_reward = 0.0


def reset_environment(task_id):
    global env, current_task, history, done, total_reward
    current_task = task_id
    env = make_env(task_id, seed=42)
    state = env.reset()
    history = []
    done = False
    total_reward = 0.0

    history.append({
        "hour": 0,
        "hr": round(state.heart_rate, 1),
        "bp": round(state.systolic_bp, 1),
        "lactate": round(state.lactate, 2),
        "sofa": state.sofa_score,
        "map": round(state.mean_arterial_pressure, 1),
        "spo2": round(state.spo2, 1),
        "temp": round(state.temperature, 1),
        "reward": 0.0,
        "status": "ADMITTED",
        "antibiotics": state.antibiotic_type.value,
        "vasopressors": state.on_vasopressors,
        "fluids_total": state.total_fluids_given_ml,
    })

    task_info = TASKS[task_id]
    info_text = (
        "TASK: " + task_info["name"] + "\n" +
        task_info["description"] + "\n\n" +
        "HINTS:\n" + "\n".join("  - " + h for h in task_info["hints"])
    )

    return (
        format_vitals(state),
        format_history(history),
        info_text,
        "Environment reset! Make your first treatment decision.",
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def take_action(fluids, antibiotic, start_vaso, stop_vaso, order_labs, escalate):
    global env, history, done, total_reward

    if env is None:
        return "Please reset the environment first!", format_history(history), "", ""

    if done:
        return "Episode is over! Please reset.", format_history(history), "", ""

    action = ClinicalAction(
        give_fluids=FluidAmount(int(fluids)),
        antibiotic=AntibioticChoice(antibiotic),
        start_vasopressors=start_vaso,
        stop_vasopressors=stop_vaso,
        order_labs=order_labs,
        escalate_care=escalate,
    )

    result = env.step(action)
    state = result.state
    total_reward += result.reward
    done = result.done

    if not state.is_alive:
        status = "DIED"
    elif state.is_stable:
        status = "STABLE"
    elif state.sofa_score > 8:
        status = "CRITICAL"
    elif state.sofa_score > 5:
        status = "SERIOUS"
    else:
        status = "UNSTABLE"

    history.append({
        "hour": len(history),
        "hr": round(state.heart_rate, 1),
        "bp": round(state.systolic_bp, 1),
        "lactate": round(state.lactate, 2),
        "sofa": state.sofa_score,
        "map": round(state.mean_arterial_pressure, 1),
        "spo2": round(state.spo2, 1),
        "temp": round(state.temperature, 1),
        "reward": round(result.reward, 3),
        "status": status,
        "antibiotics": state.antibiotic_type.value,
        "vasopressors": state.on_vasopressors,
        "fluids_total": state.total_fluids_given_ml,
    })

    feedback = (
        "Hour " + str(len(history)-1) +
        " | Reward: " + str(round(result.reward, 3)) +
        " | Total: " + str(round(total_reward, 3)) +
        " | Status: " + status
    )

    if done:
        episode_result = env.get_episode_result()
        final_score = grade(episode_result)
        feedback += (
            "\n\nEPISODE COMPLETE!" +
            "\nFinal Score: " + str(final_score) + " / 1.000" +
            "\nSurvived: " + ("Yes" if episode_result.survived else "No") +
            "\nStabilized: " + ("Yes" if episode_result.hours_to_stabilize else "No") +
            "\nFinal SOFA: " + str(episode_result.final_sofa_score) +
            "\nTotal Fluids: " + str(episode_result.total_fluids_ml) + " ml"
        )

    return (
        format_vitals(state),
        format_history(history),
        feedback,
        format_score_bar(total_reward),
    )


def run_random_agent(task_id):
    global env, history, done, total_reward
    rng = random.Random(42)
    env = make_env(task_id, seed=42)
    state = env.reset()
    history = []
    done = False
    total_reward = 0.0

    history.append({
        "hour": 0,
        "hr": round(state.heart_rate, 1),
        "bp": round(state.systolic_bp, 1),
        "lactate": round(state.lactate, 2),
        "sofa": state.sofa_score,
        "map": round(state.mean_arterial_pressure, 1),
        "spo2": round(state.spo2, 1),
        "temp": round(state.temperature, 1),
        "reward": 0.0,
        "status": "ADMITTED",
        "antibiotics": state.antibiotic_type.value,
        "vasopressors": state.on_vasopressors,
        "fluids_total": state.total_fluids_given_ml,
    })

    while not done:
        action = ClinicalAction(
            give_fluids=rng.choice(list(FluidAmount)),
            antibiotic=rng.choice(list(AntibioticChoice)),
            start_vasopressors=rng.choice([True, False]),
            stop_vasopressors=rng.choice([True, False]),
            order_labs=rng.choice([True, False]),
            escalate_care=rng.choice([True, False]),
        )
        result = env.step(action)
        state = result.state
        total_reward += result.reward
        done = result.done

        status = ("DIED" if not state.is_alive else
                  "STABLE" if state.is_stable else
                  "CRITICAL" if state.sofa_score > 8 else
                  "SERIOUS" if state.sofa_score > 5 else "UNSTABLE")

        history.append({
            "hour": len(history),
            "hr": round(state.heart_rate, 1),
            "bp": round(state.systolic_bp, 1),
            "lactate": round(state.lactate, 2),
            "sofa": state.sofa_score,
            "map": round(state.mean_arterial_pressure, 1),
            "spo2": round(state.spo2, 1),
            "temp": round(state.temperature, 1),
            "reward": round(result.reward, 3),
            "status": status,
            "antibiotics": state.antibiotic_type.value,
            "vasopressors": state.on_vasopressors,
            "fluids_total": state.total_fluids_given_ml,
        })

    episode_result = env.get_episode_result()
    final_score = grade(episode_result)

    summary = (
        "RANDOM AGENT COMPLETE\n" +
        "Task: " + task_id.upper() + "\n" +
        "Score: " + str(final_score) + " / 1.000\n" +
        "Survived: " + ("Yes" if episode_result.survived else "No") + "\n" +
        "Stabilized: " + ("Yes" if episode_result.hours_to_stabilize else "No") + "\n" +
        "Final SOFA: " + str(episode_result.final_sofa_score) + "\n" +
        "Antibiotic Changes: " + str(episode_result.antibiotic_changes) + "\n" +
        "Total Fluids: " + str(episode_result.total_fluids_ml) + " ml\n" +
        "Total Reward: " + str(round(total_reward, 3))
    )

    return (
        format_vitals(state),
        format_history(history),
        summary,
        format_score_bar(total_reward),
    )


def run_llm_agent(task_id):
    global env, history, done, total_reward

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set!", format_history(history), "Set OPENAI_API_KEY to run LLM agent.", ""

    client = OpenAI(api_key=api_key)
    env = make_env(task_id, seed=42)
    state = env.reset()
    history = []
    done = False
    total_reward = 0.0
    steps = 0
    vital_history = []

    history.append({
        "hour": 0,
        "hr": round(state.heart_rate, 1),
        "bp": round(state.systolic_bp, 1),
        "lactate": round(state.lactate, 2),
        "sofa": state.sofa_score,
        "map": round(state.mean_arterial_pressure, 1),
        "spo2": round(state.spo2, 1),
        "temp": round(state.temperature, 1),
        "reward": 0.0,
        "status": "ADMITTED",
        "antibiotics": state.antibiotic_type.value,
        "vasopressors": state.on_vasopressors,
        "fluids_total": state.total_fluids_given_ml,
    })

    while not done:
        # Track vital history for memory
        vital_history.append({
            "hour": steps,
            "hr": round(state.heart_rate, 1),
            "bp": round(state.systolic_bp, 1),
            "map": round(state.mean_arterial_pressure, 1),
            "lactate": round(state.lactate, 2),
            "sofa": state.sofa_score,
            "fluids": state.total_fluids_given_ml,
            "antibiotics": state.antibiotic_type.value,
        })
        if len(vital_history) > 3:
            vital_history.pop(0)

        # Build trend summary
        trend = ""
        if len(vital_history) >= 2:
            prev = vital_history[-2]
            curr_v = vital_history[-1]
            lac_trend = "FALLING" if curr_v["lactate"] < prev["lactate"] else "RISING"
            bp_trend = "IMPROVING" if curr_v["bp"] > prev["bp"] else "WORSENING"
            sofa_trend = "IMPROVING" if curr_v["sofa"] < prev["sofa"] else "STABLE/WORSE"
            trend = (
                "\nTRENDS:\n" +
                "- Lactate: " + lac_trend + " (" + str(prev["lactate"]) + " -> " + str(curr_v["lactate"]) + ")\n" +
                "- BP: " + bp_trend + " (" + str(prev["bp"]) + " -> " + str(curr_v["bp"]) + ")\n" +
                "- SOFA: " + sofa_trend + " (" + str(prev["sofa"]) + " -> " + str(curr_v["sofa"]) + ")\n"
            )

        history_str = ""
        if len(vital_history) >= 2:
            history_str = "\nPATIENT HISTORY (last 3 hours):\n"
            for h in vital_history[:-1]:
                history_str += (
                    "  Hour " + str(h["hour"]) + ": HR=" + str(h["hr"]) +
                    " BP=" + str(h["bp"]) + " Lactate=" + str(h["lactate"]) +
                    " SOFA=" + str(h["sofa"]) + "\n"
                )

        prompt = (
            "You are an expert ICU physician. Respond with JSON only.\n"
            "GOAL: Stabilize the patient (SOFA score <= 2 for 4+ consecutive hours)\n\n"
            "OUTPUT FORMAT:\n"
            "{\"give_fluids\": 0, \"antibiotic\": \"none\", \"start_vasopressors\": false, "
            "\"stop_vasopressors\": false, \"order_labs\": false, \"escalate_care\": false}\n\n"
            "STRICT RULES:\n"
            "- give_fluids MUST be one of: 0, 250, 500, 1000\n"
            "- antibiotic MUST be one of: none, pip-tazo, vancomycin, meropenem\n"
            "- ALWAYS give pip-tazo if Antibiotics=none\n"
            "- STOP fluids if Fluids > 3000ml OR (Lactate < 2.0 AND BP > 90)\n"
            "- Give 500ml fluids if BP < 90 or Lactate > 4.0\n"
            "- Give 250ml fluids if BP < 100 and Fluids < 2000ml\n"
            "- Start vasopressors ONLY if MAP < 65\n"
            "- Stop vasopressors if MAP > 70 and stable\n"
            "- Switch to meropenem if Lactate RISING after 12+ hours on pip-tazo\n"
            "- If SOFA improving steadily, maintain current treatment\n" +
            history_str + trend +
            "\nCURRENT PATIENT (Hour " + str(steps) + "):\n" +
            "HR=" + str(round(state.heart_rate, 1)) +
            " BP=" + str(round(state.systolic_bp, 1)) +
            " MAP=" + str(round(state.mean_arterial_pressure, 1)) +
            " Temp=" + str(round(state.temperature, 1)) +
            " SpO2=" + str(round(state.spo2, 1)) + "\n" +
            "Lactate=" + str(round(state.lactate, 2)) +
            " SOFA=" + str(state.sofa_score) +
            " Fluids=" + str(state.total_fluids_given_ml) + "ml" +
            " Antibiotics=" + state.antibiotic_type.value +
            " Vasopressors=" + str(state.on_vasopressors)
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                seed=42,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            fluids = int(data.get("give_fluids", 0))
            if fluids not in [0, 250, 500, 1000]:
                fluids = 0
            antibiotic = data.get("antibiotic", "none")
            if antibiotic not in ["none", "pip-tazo", "vancomycin", "meropenem"]:
                antibiotic = "none"
            action = ClinicalAction(
                give_fluids=FluidAmount(fluids),
                antibiotic=AntibioticChoice(antibiotic),
                start_vasopressors=bool(data.get("start_vasopressors", False)),
                stop_vasopressors=bool(data.get("stop_vasopressors", False)),
                order_labs=bool(data.get("order_labs", False)),
                escalate_care=bool(data.get("escalate_care", False)),
            )
        except:
            action = ClinicalAction(
                give_fluids=FluidAmount.SMALL,
                antibiotic=AntibioticChoice.PIP_TAZO,
                start_vasopressors=False,
                stop_vasopressors=False,
                order_labs=False,
                escalate_care=False,
            )

        result = env.step(action)
        state = result.state
        total_reward += result.reward
        steps += 1
        done = result.done

        status = ("DIED" if not state.is_alive else
                  "STABLE" if state.is_stable else
                  "CRITICAL" if state.sofa_score > 8 else
                  "SERIOUS" if state.sofa_score > 5 else "UNSTABLE")

        history.append({
            "hour": len(history),
            "hr": round(state.heart_rate, 1),
            "bp": round(state.systolic_bp, 1),
            "lactate": round(state.lactate, 2),
            "sofa": state.sofa_score,
            "map": round(state.mean_arterial_pressure, 1),
            "spo2": round(state.spo2, 1),
            "temp": round(state.temperature, 1),
            "reward": round(result.reward, 3),
            "status": status,
            "antibiotics": state.antibiotic_type.value,
            "vasopressors": state.on_vasopressors,
            "fluids_total": state.total_fluids_given_ml,
        })

    episode_result = env.get_episode_result()
    final_score = grade(episode_result)

    summary = (
        "LLM AGENT COMPLETE\n" +
        "Task: " + task_id.upper() + "\n" +
        "Score: " + str(final_score) + " / 1.000\n" +
        "Survived: " + ("Yes" if episode_result.survived else "No") + "\n" +
        "Stabilized: " + ("Yes" if episode_result.hours_to_stabilize else "No") + "\n" +
        "Final SOFA: " + str(episode_result.final_sofa_score) + "\n" +
        "Antibiotic Changes: " + str(episode_result.antibiotic_changes) + "\n" +
        "Total Fluids: " + str(episode_result.total_fluids_ml) + " ml\n" +
        "Total Reward: " + str(round(total_reward, 3)) + "\n" +
        "vs Random Agent: 0.494 avg"
    )

    return (
        format_vitals(state),
        format_history(history),
        summary,
        format_score_bar(total_reward),
    )


# ─────────────────────────────────────────────
#  FORMATTERS
# ─────────────────────────────────────────────

def format_vitals(state):
    def flag(val, low, high):
        return "[OK]" if low <= val <= high else "[!!]"

    return (
        "PATIENT VITAL SIGNS MONITOR\n" +
        "=" * 40 + "\n" +
        "Heart Rate:       " + str(round(state.heart_rate, 1)).rjust(6) + " bpm   " + flag(state.heart_rate, 60, 100) + "\n" +
        "Systolic BP:      " + str(round(state.systolic_bp, 1)).rjust(6) + " mmHg  " + flag(state.systolic_bp, 90, 140) + "\n" +
        "MAP:              " + str(round(state.mean_arterial_pressure, 1)).rjust(6) + " mmHg  " + flag(state.mean_arterial_pressure, 65, 110) + "\n" +
        "Temperature:      " + str(round(state.temperature, 1)).rjust(6) + " C     " + flag(state.temperature, 36.0, 38.0) + "\n" +
        "SpO2:             " + str(round(state.spo2, 1)).rjust(6) + " %     " + flag(state.spo2, 95, 100) + "\n" +
        "Resp Rate:        " + str(round(state.respiratory_rate, 1)).rjust(6) + " /min  " + flag(state.respiratory_rate, 12, 20) + "\n" +
        "-" * 40 + "\n" +
        "Lactate:          " + str(round(state.lactate, 2)).rjust(6) + " mmol/L " + flag(state.lactate, 0, 2.0) + "\n" +
        "WBC:              " + str(round(state.wbc, 1)).rjust(6) + " x10^9 " + flag(state.wbc, 4, 11) + "\n" +
        "Creatinine:       " + str(round(state.creatinine, 1)).rjust(6) + " uL    " + flag(state.creatinine, 60, 110) + "\n" +
        "Platelets:        " + str(round(state.platelets, 1)).rjust(6) + " x10^9 " + flag(state.platelets, 150, 400) + "\n" +
        "-" * 40 + "\n" +
        "SOFA Score:       " + str(state.sofa_score).rjust(6) + " /24\n" +
        "Hours in ICU:     " + str(state.hours_since_admission).rjust(6) + " hrs\n" +
        "On Antibiotics:   " + ("Yes  " if state.on_antibiotics else "No   ") + " (" + state.antibiotic_type.value + ")\n" +
        "On Vasopressors:  " + ("Yes" if state.on_vasopressors else "No") + "\n" +
        "Total Fluids:     " + str(state.total_fluids_given_ml).rjust(6) + " ml\n" +
        "Status:           " + ("STABLE" if state.is_stable else "ALIVE" if state.is_alive else "DIED") + "\n"
    )


def format_history(history):
    if not history:
        return "No history yet."
    lines = ["Hour  | HR    | BP    | Lactate | SOFA | MAP   | Status"]
    lines.append("-" * 60)
    for h in history[-15:]:
        lines.append(
            str(h["hour"]).rjust(4) + "  | " +
            str(h["hr"]).rjust(5) + " | " +
            str(h["bp"]).rjust(5) + " | " +
            str(h["lactate"]).rjust(7) + " | " +
            str(h["sofa"]).rjust(4) + "  | " +
            str(h["map"]).rjust(5) + " | " +
            h["status"]
        )
    return "\n".join(lines)


def format_score_bar(total_reward):
    if not history:
        return ""
    return (
        "Cumulative Reward: " + str(round(total_reward, 3)) + "\n" +
        "Hours Elapsed: " + str(len(history)-1) + "\n" +
        "Baseline (random): 0.494 avg"
    )


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(title="Sepsis ICU OpenEnv") as demo:

    gr.Markdown("""
    # Sepsis ICU — OpenEnv Environment
    ### An AI agent learns to treat sepsis patients in an ICU setting.
    > **Step()** / **Reset()** / **State()** API | 3 Tasks | Partial Reward Signals
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Setup")
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Select Task"
            )
            task_info = gr.Textbox(label="Task Description", lines=6, interactive=False)
            reset_btn = gr.Button("Reset Environment", variant="primary")
            random_btn = gr.Button("Run Random Agent", variant="secondary")
            llm_btn = gr.Button("Run LLM Agent", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Patient Monitor")
            vitals_display = gr.Textbox(
                label="Live Vitals [OK] = Normal | [!!] = Abnormal",
                lines=18,
                interactive=False,
                value="Reset the environment to begin."
            )

    gr.Markdown("---")
    gr.Markdown("### Make a Treatment Decision")

    with gr.Row():
        fluids_input = gr.Radio(choices=["0", "250", "500", "1000"], value="0", label="IV Fluids (ml)")
        antibiotic_input = gr.Radio(choices=["none", "pip-tazo", "vancomycin", "meropenem"], value="none", label="Antibiotic")

    with gr.Row():
        vaso_start = gr.Checkbox(label="Start Vasopressors")
        vaso_stop = gr.Checkbox(label="Stop Vasopressors")
        order_labs = gr.Checkbox(label="Order Labs")
        escalate = gr.Checkbox(label="Escalate Care")

    treat_btn = gr.Button("Treat Patient (Step)", variant="primary", interactive=False)

    with gr.Row():
        feedback_box = gr.Textbox(label="Action Feedback", lines=6, interactive=False)
        score_box = gr.Textbox(label="Score Tracker", lines=4, interactive=False)

    gr.Markdown("---")
    gr.Markdown("### Patient History Log")
    history_display = gr.Textbox(label="Last 15 hours", lines=18, interactive=False)

    # ── EVENTS ──
    reset_btn.click(
        fn=reset_environment,
        inputs=[task_dropdown],
        outputs=[vitals_display, history_display, task_info, feedback_box, treat_btn, random_btn]
    )
    treat_btn.click(
        fn=take_action,
        inputs=[fluids_input, antibiotic_input, vaso_start, vaso_stop, order_labs, escalate],
        outputs=[vitals_display, history_display, feedback_box, score_box]
    )
    random_btn.click(
        fn=run_random_agent,
        inputs=[task_dropdown],
        outputs=[vitals_display, history_display, feedback_box, score_box]
    )
    llm_btn.click(
        fn=run_llm_agent,
        inputs=[task_dropdown],
        outputs=[vitals_display, history_display, feedback_box, score_box]
    )

    gr.Markdown("""
    ---
    **Observation Space:** Heart Rate, BP, MAP, Temperature, SpO2, Respiratory Rate, Lactate, WBC, Creatinine, Platelets, SOFA Score
    **Action Space:** IV Fluids (0/250/500/1000ml), Antibiotic (none/pip-tazo/vancomycin/meropenem), Vasopressors, Labs, Escalation
    **Reward:** Partial progress signals every step | Terminal +1.0 for stabilization | -2.0 for death
    """)


app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
def main():
    return app