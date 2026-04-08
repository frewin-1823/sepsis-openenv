import random
import numpy as np
from models import (
    PatientState, ClinicalAction, StepResult, EpisodeResult,
    AntibioticChoice, FluidAmount
)

# ─────────────────────────────────────────────
#  HELPER FUNCTION (used in reward)
# ─────────────────────────────────────────────

def p_needs_vasopressors(state: PatientState) -> bool:
    """Returns True if patient clinically needs vasopressors."""
    return state.mean_arterial_pressure < 65 or state.systolic_bp < 90


# ─────────────────────────────────────────────
#  SEPSIS ENVIRONMENT
# ─────────────────────────────────────────────

class SepsisEnvironment:
    """
    Simulates an ICU patient with sepsis.
    The agent acts as a clinical decision support system,
    making hourly treatment decisions to stabilize the patient.
    """

    def __init__(self, task: str = "easy", seed: int = 42):
        assert task in ("easy", "medium", "hard"), "task must be easy/medium/hard"
        self.task = task
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Internal simulation state (not directly visible to agent)
        self._patient: PatientState = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._antibiotic_changes: int = 0
        self._unnecessary_antibiotic_hours: int = 0
        self._vasopressor_hours: int = 0
        self._infection_severity: float = 0.0   # 0.0 - 1.0 internal hidden var
        self._resistant_organism: bool = False

        # Task config
        self._max_steps = self._get_max_steps()

    # ─────────────────────────────────────────
    #  CORE API
    # ─────────────────────────────────────────

    def reset(self) -> PatientState:
        """Reset environment to initial state. Returns starting observation."""
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self._step_count = 0
        self._total_reward = 0.0
        self._antibiotic_changes = 0
        self._unnecessary_antibiotic_hours = 0
        self._vasopressor_hours = 0
        self._infection_severity = self._get_initial_infection_severity()
        self._resistant_organism = (self.task == "hard" and self.rng.random() < 0.5)
        self._patient = self._generate_initial_patient()
        return self.state()

    def state(self) -> PatientState:
        """Return current patient state without changing anything."""
        return self._patient.model_copy()

    def step(self, action: ClinicalAction) -> StepResult:
        """
        Apply action, simulate one hour passing, return result.
        Returns: StepResult(state, reward, done, info)
        """
        assert self._patient is not None, "Call reset() before step()"

        prev_state = self._patient.model_copy()

        # 1. Apply treatment effects
        self._apply_action(action)

        # 2. Simulate disease progression
        self._progress_disease()

        # 3. Update clinical scores
        self._update_sofa()
        self._update_stability()

        # 4. Track stats
        self._step_count += 1
        if self._patient.on_vasopressors:
            self._vasopressor_hours += 1

        # 5. Calculate reward
        reward = self._calculate_reward(prev_state, action, self._patient)
        self._total_reward += reward

        # 6. Check if episode is done
        done, reason = self._check_done()

        info = {
            "step": self._step_count,
            "reason": reason,
            "infection_severity": round(self._infection_severity, 3),
            "resistant_organism": self._resistant_organism,
        }

        return StepResult(
            state=self.state(),
            reward=round(reward, 4),
            done=done,
            info=info
        )

    def get_episode_result(self) -> EpisodeResult:
        """Call after episode ends to get gradeable result."""
        return EpisodeResult(
            task_id=self.task,
            survived=self._patient.is_alive,
            final_sofa_score=self._patient.sofa_score,
            hours_to_stabilize=self._patient.hours_stable if self._patient.is_stable else None,
            total_fluids_ml=self._patient.total_fluids_given_ml,
            antibiotic_changes=self._antibiotic_changes,
            unnecessary_antibiotic_hours=self._unnecessary_antibiotic_hours,
            vasopressor_hours=self._vasopressor_hours,
            total_steps=self._step_count,
            final_reward_sum=round(self._total_reward, 4)
        )

    # ─────────────────────────────────────────
    #  ACTION APPLICATION
    # ─────────────────────────────────────────

    def _apply_action(self, action: ClinicalAction):
        p = self._patient

        # --- Fluids ---
        if action.give_fluids != FluidAmount.NONE:
            fluid_ml = action.give_fluids.value
            p.total_fluids_given_ml += fluid_ml
            p.fluids_this_hour_ml = fluid_ml

            # Fluids improve BP if patient is hypotensive
            if p.systolic_bp < 90:
                boost = fluid_ml * 0.02
                p.systolic_bp = min(140, p.systolic_bp + boost)
                p.mean_arterial_pressure = self._calc_map(p.systolic_bp, p.diastolic_bp)

            # Fluids reduce lactate if given early
            if p.lactate > 2.0 and p.total_fluids_given_ml < 3000:
                p.lactate = max(0.5, p.lactate - fluid_ml * 0.001)
        else:
            p.fluids_this_hour_ml = 0

        # --- Antibiotics ---
        prev_antibiotic = p.antibiotic_type
        if action.antibiotic != AntibioticChoice.NONE:
            if prev_antibiotic != action.antibiotic:
                self._antibiotic_changes += 1
            p.on_antibiotics = True
            p.antibiotic_type = action.antibiotic

            # Check effectiveness
            effective = self._is_antibiotic_effective(action.antibiotic)
            if effective:
                self._infection_severity = max(0.0, self._infection_severity - 0.08)
            else:
                self._infection_severity = max(0.0, self._infection_severity - 0.01)

            # Track unnecessary antibiotic use
            if self._infection_severity < 0.1:
                self._unnecessary_antibiotic_hours += 1
        else:
            p.on_antibiotics = False
            p.antibiotic_type = AntibioticChoice.NONE

        # --- Vasopressors ---
        if action.start_vasopressors and not p.on_vasopressors:
            p.on_vasopressors = True

        if action.stop_vasopressors and p.on_vasopressors:
            # Only allow weaning if MAP is stable
            if p.mean_arterial_pressure >= 65:
                p.on_vasopressors = False

        # Vasopressors improve MAP directly
        if p.on_vasopressors:
            p.systolic_bp = min(130, p.systolic_bp + 8)
            p.diastolic_bp = min(85, p.diastolic_bp + 4)
            p.mean_arterial_pressure = self._calc_map(p.systolic_bp, p.diastolic_bp)

    # ─────────────────────────────────────────
    #  DISEASE PROGRESSION
    # ─────────────────────────────────────────

    def _progress_disease(self):
        """Simulate what happens to the patient over 1 hour."""
        p = self._patient
        sev = self._infection_severity
        noise = self.np_rng.normal

        # --- Natural disease progression based on severity ---
        if sev > 0.5:
            # Worsening: vitals deteriorate
            p.heart_rate      = min(150, p.heart_rate + noise(2, 1))
            p.systolic_bp     = max(60, p.systolic_bp - noise(2, 1))
            p.temperature     = min(40.5, p.temperature + noise(0.1, 0.05))
            p.lactate         = min(15.0, p.lactate + noise(0.15, 0.05))
            p.respiratory_rate = min(35, p.respiratory_rate + noise(0.5, 0.2))
            p.spo2            = max(80, p.spo2 - noise(0.3, 0.1))
            p.creatinine      = min(800, p.creatinine + noise(5, 2))
            p.wbc             = p.wbc + noise(0.5, 0.2)
        elif sev > 0.2:
            # Mild worsening with some noise
            p.heart_rate      = p.heart_rate + noise(0, 2)
            p.systolic_bp     = p.systolic_bp + noise(0, 3)
            p.lactate         = max(0.5, p.lactate + noise(0.0, 0.1))
        else:
            # Recovering: vitals trend toward normal
            p.heart_rate      = max(60, p.heart_rate - noise(1, 0.5))
            p.systolic_bp     = min(120, p.systolic_bp + noise(1, 0.5))
            p.temperature     = 36.5 + (p.temperature - 36.5) * 0.9
            p.lactate         = max(0.8, p.lactate * 0.92)
            p.spo2            = min(99, p.spo2 + noise(0.2, 0.1))

        # Medium/Hard: sudden deterioration event
        if self.task in ("medium", "hard") and self._step_count == 6:
            self._infection_severity = min(1.0, self._infection_severity + 0.3)
            p.systolic_bp = max(65, p.systolic_bp - 20)
            p.lactate = min(10.0, p.lactate + 2.0)

        # Clamp all values to realistic ranges
        p.heart_rate        = float(np.clip(p.heart_rate, 30, 180))
        p.systolic_bp       = float(np.clip(p.systolic_bp, 50, 200))
        p.diastolic_bp      = float(np.clip(p.diastolic_bp, 30, 120))
        p.temperature       = float(np.clip(p.temperature, 34.0, 42.0))
        p.spo2              = float(np.clip(p.spo2, 70, 100))
        p.respiratory_rate  = float(np.clip(p.respiratory_rate, 8, 40))
        p.lactate           = float(np.clip(p.lactate, 0.3, 15.0))
        p.wbc               = float(np.clip(p.wbc, 0.5, 40.0))
        p.creatinine        = float(np.clip(p.creatinine, 40, 900))
        p.platelets         = float(np.clip(p.platelets, 10, 500))
        p.mean_arterial_pressure = self._calc_map(p.systolic_bp, p.diastolic_bp)
        p.hours_since_admission += 1

    # ─────────────────────────────────────────
    #  REWARD FUNCTION
    # ─────────────────────────────────────────

    def _calculate_reward(
        self,
        prev: PatientState,
        action: ClinicalAction,
        curr: PatientState
    ) -> float:
        reward = 0.0

        # ── POSITIVE: Clinical improvement signals ──
        if curr.sofa_score < prev.sofa_score:
            reward += 0.30   # SOFA improving = big win

        if curr.lactate < prev.lactate:
            reward += 0.20   # Lactate falling = perfusion improving

        if curr.mean_arterial_pressure >= 65 and prev.mean_arterial_pressure < 65:
            reward += 0.25   # MAP normalized — critical threshold

        if curr.systolic_bp > prev.systolic_bp and prev.systolic_bp < 90:
            reward += 0.10   # BP recovering from hypotension

        if curr.spo2 >= 95 and prev.spo2 < 95:
            reward += 0.10   # Oxygen improving

        if curr.is_stable:
            reward += 0.15   # Patient stable this hour

        # ── NEGATIVE: Clinical deterioration signals ──
        if curr.sofa_score > prev.sofa_score:
            reward -= 0.25

        if curr.lactate > prev.lactate and curr.lactate > 4.0:
            reward -= 0.20   # Lactate dangerously high

        if curr.mean_arterial_pressure < 65:
            reward -= 0.20   # Septic shock range

        if curr.spo2 < 90:
            reward -= 0.15   # Dangerous hypoxia

        # ── PENALTIES: Bad clinical decisions ──
        if action.give_fluids == FluidAmount.LARGE and curr.total_fluids_given_ml > 5000:
            reward -= 0.15   # Fluid overload risk

        if self._unnecessary_antibiotic_hours > 6:
            reward -= 0.10   # Antibiotic stewardship

        if not p_needs_vasopressors(curr) and action.start_vasopressors:
            reward -= 0.10   # Unnecessary vasopressors

        # ── TIME PENALTY: Encourages efficiency ──
        reward -= 0.01

        # ── TERMINAL REWARDS ──
        if not curr.is_alive:
            reward -= 2.0

        if curr.is_stable and curr.hours_stable >= 4:
            reward += 1.0    # Sustained stabilization

        return reward

    # ─────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────

    def _check_done(self) -> tuple[bool, str]:
        p = self._patient

        if not p.is_alive:
            return True, "patient_died"

        if p.is_stable and p.hours_stable >= 4:
            return True, "patient_stabilized"

        if self._step_count >= self._max_steps:
            return True, "max_steps_reached"

        return False, "ongoing"

    def _update_sofa(self):
        """Recalculate SOFA score from current vitals."""
        p = self._patient
        score = 0

        # Respiratory (SpO2 proxy)
        if p.spo2 < 90: score += 3
        elif p.spo2 < 94: score += 2
        elif p.spo2 < 96: score += 1

        # Cardiovascular (MAP)
        if p.mean_arterial_pressure < 65: score += 3
        elif p.mean_arterial_pressure < 70: score += 1

        # Renal (Creatinine)
        if p.creatinine > 440: score += 3
        elif p.creatinine > 170: score += 2
        elif p.creatinine > 110: score += 1

        # Coagulation (Platelets)
        if p.platelets < 50: score += 3
        elif p.platelets < 100: score += 2
        elif p.platelets < 150: score += 1

        # Lactate as liver proxy
        if p.lactate > 8.0: score += 3
        elif p.lactate > 4.0: score += 2
        elif p.lactate > 2.0: score += 1

        p.sofa_score = min(24, score)
        p.is_alive = (p.sofa_score < 20 and p.mean_arterial_pressure > 40)

    def _update_stability(self):
        p = self._patient
        vitals_normal = (
            60 <= p.heart_rate <= 100 and
            90 <= p.systolic_bp <= 140 and
            p.mean_arterial_pressure >= 65 and
            36.0 <= p.temperature <= 38.0 and
            p.spo2 >= 95 and
            p.lactate < 2.0 and
            p.sofa_score <= 2
        )
        if vitals_normal:
            p.hours_stable += 1
            p.is_stable = True
        else:
            p.hours_stable = 0
            p.is_stable = False

    def _is_antibiotic_effective(self, antibiotic: AntibioticChoice) -> bool:
        if self._resistant_organism:
            return antibiotic == AntibioticChoice.MEROPENEM
        return antibiotic in (AntibioticChoice.PIP_TAZO, AntibioticChoice.VANCOMYCIN)

    def _calc_map(self, sbp: float, dbp: float) -> float:
        return round((sbp + 2 * dbp) / 3, 1)

    def _get_max_steps(self) -> int:
        return {"easy": 36, "medium": 48, "hard": 72}[self.task]
        
    def _get_initial_infection_severity(self) -> float:
        return {"easy": 0.35, "medium": 0.55, "hard": 0.70}[self.task]

    def _generate_initial_patient(self) -> PatientState:
        sev = self._infection_severity
        rng = self.np_rng

        sbp = float(np.clip(rng.normal(95 - sev * 20, 5), 60, 130))
        dbp = float(np.clip(rng.normal(60 - sev * 10, 3), 40, 85))

        return PatientState(
            heart_rate=float(np.clip(rng.normal(100 + sev * 20, 5), 60, 150)),
            systolic_bp=sbp,
            diastolic_bp=dbp,
            mean_arterial_pressure=self._calc_map(sbp, dbp),
            temperature=float(np.clip(rng.normal(38.5 + sev * 0.8, 0.3), 36.0, 41.0)),
            spo2=float(np.clip(rng.normal(94 - sev * 5, 1), 80, 99)),
            respiratory_rate=float(np.clip(rng.normal(22 + sev * 6, 2), 12, 35)),
            lactate=float(np.clip(rng.normal(2.5 + sev * 4, 0.5), 0.8, 12.0)),
            wbc=float(np.clip(rng.normal(14 + sev * 6, 2), 2, 35)),
            creatinine=float(np.clip(rng.normal(120 + sev * 80, 15), 60, 500)),
            platelets=float(np.clip(rng.normal(180 - sev * 60, 20), 30, 400)),
            sofa_score=int(np.clip(round(sev * 10), 0, 15)),
            hours_since_admission=0,
            on_vasopressors=False,
            on_antibiotics=False,
            antibiotic_type=AntibioticChoice.NONE,
            total_fluids_given_ml=0,
            fluids_this_hour_ml=0,
            is_alive=True,
            is_stable=False,
            hours_stable=0
        )



