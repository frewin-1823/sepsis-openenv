    ---
    title: Sepsis ICU OpenEnv
    emoji: 🏥
    colorFrom: red
    colorTo: blue
    sdk: docker
    pinned: false
    ---
    # Sepsis ICU — OpenEnv Environment

    An AI agent learns to treat sepsis patients in an ICU setting.
    The agent acts as a clinical decision support system, making hourly
    treatment decisions to stabilize critically ill patients.

    Sepsis kills over 11 million people per year worldwide. Early and
    correct treatment decisions dramatically improve survival. This
    environment challenges an AI agent to learn those decisions from
    scratch through reinforcement learning.

    ---

    ## Environment Description

    At every timestep (1 hour), the agent observes a patient's full
    clinical picture — vital signs, lab values, and treatment status —
    and must decide what interventions to apply. The environment simulates
    realistic sepsis progression, antibiotic resistance, and multi-organ
    dysfunction based on the Surviving Sepsis Campaign guidelines.

    ---

    ## Observation Space

    ### Vital Signs
    | Field | Type | Range | Unit | Normal |
    |---|---|---|---|---|
    | heart_rate | float | 30-180 | bpm | 60-100 |
    | systolic_bp | float | 50-200 | mmHg | 90-140 |
    | diastolic_bp | float | 30-120 | mmHg | 60-90 |
    | mean_arterial_pressure | float | 30-150 | mmHg | >65 |
    | temperature | float | 34.0-42.0 | celsius | 36.5-37.5 |
    | spo2 | float | 70-100 | percent | >95 |
    | respiratory_rate | float | 8-40 | breaths/min | 12-20 |

    ### Lab Values
    | Field | Type | Range | Unit | Normal |
    |---|---|---|---|---|
    | lactate | float | 0.3-15.0 | mmol/L | <2.0 |
    | wbc | float | 0.5-40.0 | x10^9/L | 4-11 |
    | creatinine | float | 40-900 | umol/L | 60-110 |
    | platelets | float | 10-500 | x10^9/L | 150-400 |

    ### Clinical Status
    | Field | Type | Description |
    |---|---|---|
    | sofa_score | int (0-24) | Organ failure severity. Higher = worse |
    | hours_since_admission | int | Time elapsed in ICU |
    | on_vasopressors | bool | Blood pressure support active |
    | on_antibiotics | bool | Antibiotic treatment active |
    | antibiotic_type | str | none / pip-tazo / vancomycin / meropenem |
    | total_fluids_given_ml | int | Cumulative IV fluids given |
    | is_alive | bool | Patient survival status |
    | is_stable | bool | All vitals within normal range |

    ---

    ## Action Space

    | Action | Type | Values | Description |
    |---|---|---|---|
    | give_fluids | int | 0, 250, 500, 1000 (ml) | IV fluid bolus this hour |
    | antibiotic | str | none, pip-tazo, vancomycin, meropenem | Antibiotic to prescribe |
    | start_vasopressors | bool | True/False | Start norepinephrine drip |
    | stop_vasopressors | bool | True/False | Wean off vasopressors |
    | order_labs | bool | True/False | Order new blood work |
    | escalate_care | bool | True/False | Call senior physician |

    ---

    ## Reward Function

    Partial progress signals are given at every timestep so the agent
    receives feedback throughout the episode, not just at the end.

    ### Positive Rewards (per step)
    - SOFA score decreasing: +0.30
    - Lactate falling: +0.20
    - MAP crossing above 65 mmHg: +0.25
    - BP recovering from hypotension: +0.10
    - SpO2 returning above 95%: +0.10
    - Patient stable this hour: +0.15

    ### Negative Rewards (per step)
    - SOFA score increasing: -0.25
    - Lactate dangerously high (>4.0): -0.20
    - MAP below 65 (septic shock): -0.20
    - SpO2 below 90%: -0.15
    - Fluid overload (>5000ml total): -0.15
    - Antibiotic overuse: -0.10
    - Unnecessary vasopressors: -0.10
    - Time penalty per step: -0.01

    ### Terminal Rewards
    - Patient sustained stabilization (4+ hours): +1.0
    - Patient death: -2.0

    ---

    ## Tasks

    ### Easy — Stable Sepsis (UTI Source)
    Single patient with mild sepsis from a urinary tract infection.
    Standard antibiotics are effective. Vitals mildly abnormal.
    Goal: Stabilize within 24 hours.
    Max steps: 24

    ### Medium — Deteriorating Sepsis (Pneumonia Source)
    Patient deteriorates sharply at hour 6. Requires vasopressors and
    aggressive management. Agent must anticipate and respond quickly.
    Goal: Prevent septic shock, stabilize within 48 hours.
    Max steps: 48

    ### Hard — Multi-System Sepsis (Resistant Organism)
    Severe sepsis with 50% chance of antibiotic-resistant organism
    requiring meropenem. Multiple deterioration events. Limited time.
    Goal: Maximize survival and minimize organ damage over 72 hours.
    Max steps: 72

    ---

    ## Agent Graders (Score 0.0 to 1.0)

    ### Easy Grader
    - Survival: 0.40
    - Stabilization: 0.30
    - SOFA improvement: 0.20
    - Efficiency (speed): 0.10

    ### Medium Grader
    - Survival: 0.35
    - Stabilization: 0.25
    - SOFA improvement: 0.20
    - Antibiotic stewardship: 0.10
    - Vasopressor management: 0.10

    ### Hard Grader
    - Survival: 0.30
    - Stabilization: 0.20
    - SOFA improvement: 0.15
    - Antibiotic reasoning: 0.15
    - Stewardship: 0.10
    - Fluid management: 0.10

    ---

    ## Baseline Scores (Random Agent, Seed 42)

    | Task | Score | Survived | Steps |
    |---|---|---|---|
    | Easy | 0.540 | Yes | 24 |
    | Medium | 0.533 | Yes | 48 |
    | Hard | 0.410 | Yes | 72 |
    | **Average** | **0.494** | | |

    A well-trained AI agent should score above 0.494 average.

    ---

    ## API Usage

    ```python
    from environment import SepsisEnvironment
    from models import ClinicalAction, FluidAmount, AntibioticChoice

    # Create environment
    env = SepsisEnvironment(task="easy", seed=42)

    # Reset
    state = env.reset()

    # Step
    action = ClinicalAction(
        give_fluids=FluidAmount.MEDIUM,
        antibiotic=AntibioticChoice.PIP_TAZO,
        start_vasopressors=False,
        stop_vasopressors=False,
        order_labs=True,
        escalate_care=False,
    )
    result = env.step(action)
    print(result.state, result.reward, result.done)

    # Grade at end
    episode_result = env.get_episode_result()
    from tasks import grade
    score = grade(episode_result)
    print("Score:", score)
    ```

    ---

    ## Setup Instructions

    ### Local Setup
    ```bash
    git clone https://huggingface.co/spaces/YOUR_USERNAME/sepsis-openenv
    cd sepsis-openenv
    pip install -r requirements.txt
    ```

    ### Run Baseline
    ```bash
    python inference.py
    ```

    ### Run Interactive UI
    ```bash
    python app.py
    # Open http://localhost:7860
    ```

    ### Run with Docker
    ```bash
    docker build -t sepsis-openenv .
    docker run -p 7860:7860 sepsis-openenv
    # Open http://localhost:7860
    ```

    ---

    ## Project Structure

    ```
    sepsis-openenv/
    ├── models.py         Typed Pydantic models (State, Action, Result)
    ├── environment.py    Core environment (reset, step, state)
    ├── tasks.py          Task configs and grader functions
    ├── baseline.py       Random agent baseline with reproducible scores
    ├── app.py            Gradio interactive UI
    ├── openenv.yaml      Full OpenEnv specification
    ├── Dockerfile        Container setup
    ├── requirements.txt  Python dependencies
    └── README.md         This file
    ```

    ---

    ## Clinical Grounding

    This environment is designed around real clinical guidelines:

    - Surviving Sepsis Campaign (SSC) 2021 Guidelines
    - SOFA score for organ failure assessment
    - MAP greater than 65 mmHg as vasopressor threshold
    - Lactate clearance as treatment response marker
    - Antibiotic stewardship principles
    - Conservative fluid resuscitation strategy

    ---

    ## License

    MIT License
