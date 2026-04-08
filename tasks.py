from environment import SepsisEnvironment
from models import EpisodeResult


# ─────────────────────────────────────────────
#  TASK DEFINITIONS
# ─────────────────────────────────────────────

TASKS = {
    "easy": {
        "id": "easy",
        "name": "Stable Sepsis — UTI Source",
        "description": (
            "A 58-year-old patient presents with sepsis from a urinary tract infection. "
            "Vitals are mildly abnormal. Infection responds to standard antibiotics. "
            "Goal: Stabilize the patient within 24 hours."
        ),
        "max_steps": 36,
        "seed": 42,
        "hints": [
            "Start antibiotics early — pip-tazo works here",
            "Give fluids if BP is low",
            "Watch lactate — it should fall with treatment",
        ]
    },
    "medium": {
        "id": "medium",
        "name": "Deteriorating Sepsis — Pneumonia Source",
        "description": (
            "A 67-year-old patient with sepsis from pneumonia. "
            "Initially manageable, but deteriorates sharply at hour 6. "
            "Requires vasopressors and aggressive management. "
            "Goal: Prevent septic shock, stabilize within 48 hours."
        ),
        "max_steps": 48,
        "seed": 42,
        "hints": [
            "Watch for the crash at hour 6 — be ready to escalate",
            "Start vasopressors if MAP drops below 65",
            "Vancomycin may be needed for pneumonia coverage",
        ]
    },
    "hard": {
        "id": "hard",
        "name": "Multi-Patient ICU — Resistant Organism",
        "description": (
            "A critically ill patient with severe sepsis. "
            "50% chance of antibiotic-resistant organism (requires meropenem). "
            "Multiple deterioration events. Limited resources. "
            "Goal: Maximize survival and minimize organ damage over 72 hours."
        ),
        "max_steps": 72,
        "seed": 42,
        "hints": [
            "If standard antibiotics aren't working, switch to meropenem",
            "Monitor creatinine — kidney failure is common here",
            "Escalate care early if SOFA score exceeds 10",
        ]
    }
}


# ─────────────────────────────────────────────
#  GRADER FUNCTIONS (score 0.0 → 1.0)
# ─────────────────────────────────────────────

def grade_easy(result: EpisodeResult) -> float:
    """
    Easy grader — focuses on basic survival and stabilization.
    Score breakdown:
      0.4 — Did the patient survive?
      0.3 — Was the patient stabilized?
      0.2 — Was lactate/BP managed? (proxy: final SOFA score)
      0.1 — Efficiency (did it happen quickly?)
    """
    score = 0.0

    # Survival (0.4)
    if result.survived:
        score += 0.4

    # Stabilization (0.3)
    if result.hours_to_stabilize is not None:
        score += 0.3

    # Low final SOFA score (0.2)
    sofa_score = max(0, 1.0 - result.final_sofa_score / 10.0)
    score += 0.2 * sofa_score

    # Efficiency — stabilized quickly (0.1)
    if result.hours_to_stabilize is not None:
        efficiency = 1.0 - (result.hours_to_stabilize / 24.0)
        score += 0.1 * max(0.0, efficiency)

    return round(min(1.0, score), 3)


def grade_medium(result: EpisodeResult) -> float:
    """
    Medium grader — adds antibiotic stewardship and vasopressor management.
    Score breakdown:
      0.35 — Survival
      0.25 — Stabilization
      0.20 — SOFA score improvement
      0.10 — Antibiotic stewardship (no overuse)
      0.10 — Appropriate vasopressor use
    """
    score = 0.0

    # Survival (0.35)
    if result.survived:
        score += 0.35

    # Stabilization (0.25)
    if result.hours_to_stabilize is not None:
        score += 0.25

    # SOFA score (0.20)
    sofa_score = max(0, 1.0 - result.final_sofa_score / 12.0)
    score += 0.20 * sofa_score

    # Antibiotic stewardship (0.10)
    if result.unnecessary_antibiotic_hours <= 3:
        score += 0.10
    elif result.unnecessary_antibiotic_hours <= 6:
        score += 0.05

    # Vasopressor management (0.10)
    # Penalize if vasopressors used too long unnecessarily
    if result.vasopressor_hours <= 12:
        score += 0.10
    elif result.vasopressor_hours <= 24:
        score += 0.05

    return round(min(1.0, score), 3)


def grade_hard(result: EpisodeResult) -> float:
    """
    Hard grader — full clinical decision quality assessment.
    Score breakdown:
      0.30 — Survival
      0.20 — Stabilization
      0.15 — SOFA score
      0.15 — Antibiotic choice quality (few switches = good reasoning)
      0.10 — Stewardship
      0.10 — Fluid management
    """
    score = 0.0

    # Survival (0.30)
    if result.survived:
        score += 0.30

    # Stabilization (0.20)
    if result.hours_to_stabilize is not None:
        score += 0.20

    # SOFA score (0.15)
    sofa_score = max(0, 1.0 - result.final_sofa_score / 15.0)
    score += 0.15 * sofa_score

    # Antibiotic reasoning (0.15) — fewer switches = more decisive
    if result.antibiotic_changes <= 1:
        score += 0.15
    elif result.antibiotic_changes <= 3:
        score += 0.08

    # Stewardship (0.10)
    if result.unnecessary_antibiotic_hours <= 2:
        score += 0.10
    elif result.unnecessary_antibiotic_hours <= 5:
        score += 0.05

    # Fluid management (0.10) — avoid overloading
    if result.total_fluids_ml <= 4000:
        score += 0.10
    elif result.total_fluids_ml <= 6000:
        score += 0.05

    return round(min(1.0, score), 3)


# ─────────────────────────────────────────────
#  MASTER GRADER
# ─────────────────────────────────────────────

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

def grade(result: EpisodeResult) -> float:
    """Grade an episode result. Returns score between 0.0 and 1.0."""
    grader = GRADERS.get(result.task_id)
    if not grader:
        raise ValueError(f"Unknown task: {result.task_id}")
    return grader(result)


def get_task(task_id: str) -> dict:
    """Get task config by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Choose from: easy, medium, hard")
    return TASKS[task_id]


def make_env(task_id: str, seed: int = 42) -> SepsisEnvironment:
    """Create a ready-to-use environment for a given task."""
    return SepsisEnvironment(task=task_id, seed=seed)
