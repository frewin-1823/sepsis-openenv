import random
from models import ClinicalAction, FluidAmount, AntibioticChoice
from tasks import make_env

SEED = 42

def log_start(task, env, model):
    print("[START] task={} env={} model={}".format(task, env, model))

def log_step(step, action, reward, done):
    print("[STEP] step={} action={} reward={:.2f} done={} error=null".format(
        step, action, reward, str(done).lower()
    ))

def log_end(success, steps, rewards):
    rewards_str = ",".join(["{:.2f}".format(r) for r in rewards])
    print("[END] success={} steps={} rewards={}".format(
        str(success).lower(), steps, rewards_str
    ))

def ask_rule_based(state, task_id, step):

    if state.antibiotic_type.value == "none":
        antibiotic = AntibioticChoice.PIP_TAZO
    elif task_id == "hard" and state.lactate > 3.5 and step > 10:
        antibiotic = AntibioticChoice.MEROPENEM
    elif task_id == "medium" and step > 6:
        antibiotic = AntibioticChoice.VANCOMYCIN
    else:
        antibiotic = state.antibiotic_type

    if state.total_fluids_given_ml > 4000:
        fluids = FluidAmount.NONE
    elif state.systolic_bp < 85 or state.lactate > 4.5:
        fluids = FluidAmount.LARGE
    elif state.systolic_bp < 95 or state.lactate > 3.0:
        fluids = FluidAmount.MEDIUM
    elif state.systolic_bp < 105:
        fluids = FluidAmount.SMALL
    else:
        fluids = FluidAmount.NONE

    start_vaso = state.mean_arterial_pressure < 65
    stop_vaso = state.mean_arterial_pressure > 70

    escalate = (task_id == "hard" and state.sofa_score > 10)

    return ClinicalAction(
        give_fluids=fluids,
        antibiotic=antibiotic,
        start_vasopressors=start_vaso,
        stop_vasopressors=stop_vaso,
        order_labs=False,
        escalate_care=escalate,
    )

def run_task(task_id):
    env = make_env(task_id, seed=SEED)
    state = env.reset()

    rewards = []
    step = 0
    done = False

    log_start(task_id, "sepsis", "rule-based")

    while not done:
        step += 1

        action = ask_rule_based(state, task_id, step)
        result = env.step(action)

        reward = result.reward if result.reward else 0.0
        done = result.done

        rewards.append(reward)

        action_str = "fluids={},abx={},vaso={}".format(
            action.give_fluids.name,
            action.antibiotic.name,
            action.start_vasopressors
        )

        log_step(step, action_str, reward, done)

        state = result.state

    success = rewards[-1] > 0 if rewards else False
    log_end(success, step, rewards)

if __name__ == "__main__":
    random.seed(SEED)

    for task in ["easy", "medium", "hard"]:
        run_task(task)