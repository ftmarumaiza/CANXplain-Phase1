"""Ablation arm definitions.

Each arm removes exactly ONE thing from A0 (except A5, which removes two
deliberately, to give a conventional end-to-end baseline). Everything
else -- data, splits, folds, seeds, metrics, evaluation code -- is held
identical across arms, so any difference in outcome is attributable to
the removed component.

The `question` field on each arm is not decoration. It is the claim the
arm licenses, and it is what gets printed into docs/ABLATION.md and the
results summary. If an arm's result does not answer its question, the
arm is the wrong experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..features.extractor import DEFAULT_GROUPS
from ..meta.fitness import FitnessSpec


@dataclass
class AblationArm:
    code: str
    name: str
    removes: str
    question: str
    use_meta_learning: bool = True
    use_shap_consistency: bool = True
    use_complexity: bool = True
    feature_groups: Optional[List[str]] = None       # None = config default
    notes: str = ""

    def spec(self, base: FitnessSpec) -> FitnessSpec:
        """Objective for this arm, derived from the configured base."""
        return base.variant(
            use_shap_consistency=self.use_shap_consistency,
            use_complexity=self.use_complexity,
        )

    def groups(self, default: List[str]) -> List[str]:
        return list(self.feature_groups) if self.feature_groups else list(default)

    def as_record(self) -> Dict[str, object]:
        return {
            "arm": self.code,
            "arm_name": self.name,
            "arm_description": self.removes,
            "arm_question": self.question,
            "arm_uses_meta_learning": self.use_meta_learning,
            "arm_uses_shap_consistency": self.use_shap_consistency,
            "arm_uses_complexity": self.use_complexity,
        }


ARMS: Dict[str, AblationArm] = {
    "A0": AblationArm(
        code="A0",
        name="Full CANXplain",
        removes="nothing -- the complete proposed system",
        question=(
            "What does the full system achieve, same-dataset and "
            "cross-dataset? This is the reference every other arm is "
            "compared against."
        ),
        use_meta_learning=True,
        use_shap_consistency=True,
        use_complexity=True,
        notes=("Vendor-agnostic temporal + frequency features, RF/DNN "
               "candidates, ACHILLES-inspired search, explanation-aware "
               "complexity-penalised fitness."),
    ),
    "A1": AblationArm(
        code="A1",
        name="Without meta-learning",
        removes="the ACHILLES-inspired competition/evolution search",
        question=(
            "How much of the result comes from searching the "
            "configuration space at all, as opposed to the features and "
            "the objective? If A1 matches A0, the search is not earning "
            "its compute and should be dropped."
        ),
        use_meta_learning=False,
        use_shap_consistency=True,
        use_complexity=True,
        notes=("Fixed default RF and DNN configurations, scored under the "
               "same fitness function; the better of the two is selected."),
    ),
    "A2": AblationArm(
        code="A2",
        name="Accuracy-only selection",
        removes="explanation-awareness and the complexity penalty from selection",
        question=(
            "Does explanation-aware selection buy anything over "
            "conventional accuracy-driven model selection? This is the "
            "single most important ablation in the study -- it is the "
            "direct test of the CANXplain premise, and it is the arm "
            "that most closely reproduces ACHILLES-style selection."
        ),
        use_meta_learning=True,
        use_shap_consistency=False,
        use_complexity=False,
        notes=("Search is retained; the objective becomes Fitness = "
               "Accuracy. SHAP consistency is still MEASURED for the "
               "selected model so A0 and A2 can be compared on it, but it "
               "does not influence the search."),
    ),
    "A3": AblationArm(
        code="A3",
        name="Without SHAP consistency",
        removes="the beta * SHAP_consistency term only",
        question=(
            "Isolated from the complexity penalty, what does explanation "
            "stability contribute? A2 removes two things at once; A3 "
            "removes only this one, so the difference between A2 and A3 "
            "attributes the effect correctly."
        ),
        use_meta_learning=True,
        use_shap_consistency=False,
        use_complexity=True,
        notes="Fitness = alpha*Accuracy - gamma*Complexity.",
    ),
    "A4": AblationArm(
        code="A4",
        name="Without complexity penalty",
        removes="the gamma * Complexity term only",
        question=(
            "Does complexity-aware selection actually restrain model "
            "size, and at what cost in detection performance? If A4 "
            "selects far larger models for no accuracy gain, the penalty "
            "is doing useful work for on-vehicle deployment."
        ),
        use_meta_learning=True,
        use_shap_consistency=True,
        use_complexity=False,
        notes="Fitness = alpha*Accuracy + beta*SHAP_consistency.",
    ),
    "A5": AblationArm(
        code="A5",
        name="No meta-learning + accuracy-only",
        removes="both the search and the explanation/complexity terms",
        question=(
            "What does a conventional practitioner pipeline produce on "
            "exactly this data and these splits? This is the honest "
            "floor: if A0 cannot beat A5, the framework has not "
            "justified itself."
        ),
        use_meta_learning=False,
        use_shap_consistency=False,
        use_complexity=False,
        notes="Default RF and DNN, selected on accuracy alone.",
    ),
    "A6": AblationArm(
        code="A6",
        name="Reduced feature set",
        removes="the proposed temporal and frequency feature engineering",
        question=(
            "How much of the performance comes from the proposed "
            "vendor-agnostic features rather than from the model "
            "machinery? Everything else is held at A0 settings."
        ),
        feature_groups=["basic"],
        notes=("Minimal conventional window descriptors only: message "
               "count, window duration, unique ID count."),
    ),
    "A7": AblationArm(
        code="A7",
        name="Temporal features only",
        removes="the frequency / CAN-ID statistics groups",
        question=(
            "Do timing features alone generalise across vehicles? Timing "
            "is the most vendor-neutral signal available, so if A7 holds "
            "up cross-dataset it is the strongest argument for the "
            "vendor-agnostic premise."
        ),
        feature_groups=["temporal", "temporal_id"],
    ),
    "A8": AblationArm(
        code="A8",
        name="Frequency features only",
        removes="the timing-based feature groups",
        question=(
            "Do CAN-ID frequency statistics alone generalise? Compared "
            "with A7 this shows which of the two feature families "
            "carries cross-dataset transfer, and whether A0 needs both."
        ),
        feature_groups=["frequency"],
    ),
}

DEFAULT_ARM_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

# The subset shown in the mandatory ablation figure (section 15 of the spec).
FIGURE_ARMS = ["A0", "A1", "A2", "A3", "A4", "A6"]


def get_arms(codes: Optional[List[str]] = None) -> List[AblationArm]:
    codes = codes or DEFAULT_ARM_ORDER
    unknown = [c for c in codes if c not in ARMS]
    if unknown:
        raise ValueError(f"unknown ablation arm(s) {unknown}; available: {DEFAULT_ARM_ORDER}")
    return [ARMS[c] for c in codes]


def ablation_documentation() -> str:
    """Markdown describing what each arm removes and why -- docs/ABLATION.md."""
    lines = [
        "# Ablation study: what each arm removes and what it answers",
        "",
        "All arms share identical data, chronological splits, CV folds, "
        "seeds, metrics and evaluation code. Only the named component "
        "changes. Results for every arm are reported both same-dataset "
        "and cross-dataset; the cross-dataset numbers are the ones that "
        "bear on the generalisation claim.",
        "",
    ]
    for code in DEFAULT_ARM_ORDER:
        arm = ARMS[code]
        lines += [
            f"## {arm.code} — {arm.name}",
            "",
            f"**Removes:** {arm.removes}",
            "",
            f"**Scientific question:** {arm.question}",
            "",
            f"**Selection objective:** "
            f"{'meta-learning search' if arm.use_meta_learning else 'fixed default configurations'}, "
            f"SHAP consistency {'in' if arm.use_shap_consistency else 'not in'} the objective, "
            f"complexity penalty {'in' if arm.use_complexity else 'not in'} the objective.",
            "",
            f"**Feature groups:** "
            f"{', '.join(arm.feature_groups) if arm.feature_groups else 'config default (' + ', '.join(DEFAULT_GROUPS) + ')'}",
            "",
        ]
        if arm.notes:
            lines += [f"**Notes:** {arm.notes}", ""]
    lines += [
        "## Reading the results honestly",
        "",
        "- A difference between two arms on a single seed is not a result. "
        "Check the seed-level standard deviation in Table 6 first.",
        "- The paired tests in `statistical_tests.csv` are underpowered "
        "below 5 seeds and are flagged as such. Do not report a p-value "
        "from 3 seeds as a significance claim.",
        "- If A0 does not beat A2 cross-dataset, the explanation-aware "
        "selection hypothesis is not supported on these datasets. That is "
        "a publishable finding; it is not a reason to retune the weights "
        "until it is.",
        "",
    ]
    return "\n".join(lines)
