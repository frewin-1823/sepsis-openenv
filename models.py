from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


# ─────────────────────────────────────────────
#  ENUMS — Fixed choices for actions
# ─────────────────────────────────────────────

class AntibioticChoice(str, Enum):
    NONE         = "none"
    PIP_TAZO     = "pip-tazo"        # broad spectrum, first line
    VANCOMYCIN   = "vancomycin"      # gram positive / MRSA
    MEROPENEM    = "meropenem"       # last resort, resistant organisms


class FluidAmount(int, Enum):
    NONE    = 0
    SMALL   = 250    # ml
    MEDIUM  = 500    # ml
    LARGE   = 1000   # ml


# ─────────────────────────────────────────────
#  PATIENT STATE — What the agent "sees"
# ─────────────────────────────────────────────

class PatientState(BaseModel):
    """
    Complete snapshot of a patient at one timestep (1 hour).
    All values are realistic clinical ranges.
    """

    # --- Vital Signs ---
    heart_rate: float = Field(..., description="Beats per minute. Normal: 60-100")
    systolic_bp: float = Field(..., description="Blood pressure mmHg. Normal: 90-140")
    diastolic_bp: float = Field(..., description="Blood pressure mmHg. Normal: 60-90")
    temperature: float = Field(..., description="Celsius. Normal: 36.5-37.5")
    spo2: float = Field(..., description="Oxygen saturation %. Normal: 95-100")
    respiratory_rate: float = Field(..., description="Breaths/min. Normal: 12-20")
    mean_arterial_pressure: float = Field(..., description="MAP mmHg. Critical if < 65")

    # --- Lab Values ---
    lactate: float = Field(..., description="mmol/L. Normal < 2.0, danger > 4.0")
    wbc: float = Field(..., description="White blood cells x10^9/L. Normal: 4-11")
    creatinine: float = Field(..., description="Kidney function umol/L. Normal: 60-110")
    platelets: float = Field(..., description="x10^9/L. Normal: 150-400")

    # --- Clinical Severity Score ---
    sofa_score: int = Field(..., description="0-24. Higher = worse. >10 = critical")

    # --- Time ---
    hours_since_admission: int = Field(..., description="How long patient has been in ICU")

    # --- Current Treatment Status ---
    on_vasopressors: bool = Field(..., description="Is patient on BP support medication")
    on_antibiotics: bool = Field(..., description="Is patient currently on antibiotics")
    antibiotic_type: AntibioticChoice = Field(..., description="Which antibiotic is active")
    total_fluids_given_ml: int = Field(..., description="Cumulative fluids given so far")
    fluids_this_hour_ml: int = Field(..., description="Fluids given in last timestep")

    # --- Patient Outcome Status ---
    is_alive: bool = Field(default=True)
    is_stable: bool = Field(default=False)
    hours_stable: int = Field(default=0, description="Consecutive hours with normal vitals")


# ─────────────────────────────────────────────
#  CLINICAL ACTION — What the agent can do
# ─────────────────────────────────────────────

class ClinicalAction(BaseModel):
    """
    One action taken by the agent at each timestep.
    Mirrors real ICU hourly decisions.
    """

    give_fluids: FluidAmount = Field(
        default=FluidAmount.NONE,
        description="IV fluid bolus to give this hour"
    )

    antibiotic: AntibioticChoice = Field(
        default=AntibioticChoice.NONE,
        description="Which antibiotic to prescribe (none = stop/don't start)"
    )

    start_vasopressors: bool = Field(
        default=False,
        description="Start norepinephrine drip for blood pressure support"
    )

    stop_vasopressors: bool = Field(
        default=False,
        description="Wean off vasopressors if patient improving"
    )

    order_labs: bool = Field(
        default=False,
        description="Order new blood labs (costs time, gives fresh data)"
    )

    escalate_care: bool = Field(
        default=False,
        description="Call senior physician — used when agent is uncertain"
    )


# ─────────────────────────────────────────────
#  STEP RESULT — What comes back after step()
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Everything returned by environment.step()
    """
    state: PatientState
    reward: float = Field(..., description="Reward signal for this timestep")
    done: bool = Field(..., description="True if episode is over")
    info: dict = Field(default_factory=dict, description="Extra debug metadata")


# ─────────────────────────────────────────────
#  EPISODE RESULT — Final grading input
# ─────────────────────────────────────────────

class EpisodeResult(BaseModel):
    """
    Summary of a completed episode — fed into the grader.
    """
    task_id: Literal["easy", "medium", "hard"]
    survived: bool
    final_sofa_score: int
    hours_to_stabilize: Optional[int] = None   # None if never stabilized
    total_fluids_ml: int
    antibiotic_changes: int                    # how many times antibiotic was switched
    unnecessary_antibiotic_hours: int          # hours antibiotics given when not needed
    vasopressor_hours: int                     # hours patient was on vasopressors
    total_steps: int
    final_reward_sum: float
