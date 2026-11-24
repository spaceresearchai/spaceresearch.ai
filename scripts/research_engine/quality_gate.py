#!/usr/bin/env python3
"""
NeurIPS/ICLR Conference-Readiness Quality Gate (`quality_gate.py`).

Enforces a strict quantitative & methodological bar before any autonomous paper is
eligible for public Zenodo DOI publication. Investigations that do not satisfy ALL
criteria are routed strictly to private Zenodo drafts (`draft_created`) rather than
permanent public DOIs under Marcio Diaz / Space AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QualityReport:
    passed: bool
    score_10: float
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    recommended_action: str = "draft_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score_10": round(self.score_10, 2),
            "checks": self.checks,
            "reasons": self.reasons,
            "recommended_action": self.recommended_action,
        }


def evaluate_conference_readiness(exp_data: Dict[str, Any]) -> QualityReport:
    """
    Evaluate whether an experiment meets NeurIPS/ICLR/MLSys submission standards:
      1. Multi-model OR multi-seed + multi-regime statistical evaluation (not 1-shot toy)
      2. >= 3 strong published baselines compared under identical conditions
      3. Formal statistical uncertainty (95% bootstrap CIs, std across seeds, or p-values)
      4. Real non-trivial scale (e.g. >= 2 model families, or overcomplete SAE >= 4x / >= 50k tokens, or LLM vocab >= 32k + real Triton kernel + gradient eval)
      5. Explicit ablation study + honest limitations + related work citations (>= 6 references)
    """
    qg = exp_data.get("quality_evidence") or {}
    checks: Dict[str, bool] = {
        "multi_model_or_multi_seed_scale": bool(
            int(qg.get("n_models", 0)) >= 2
            or (int(qg.get("n_seeds", 0)) >= 3 and int(qg.get("n_regimes", 0)) >= 3)
        ),
        "strong_baselines_ge3": int(qg.get("n_baselines", 0)) >= 3,
        "statistical_uncertainty": bool(qg.get("has_confidence_intervals") or qg.get("has_p_values")),
        "ablation_study": bool(qg.get("has_ablations")),
        "real_world_scale": bool(qg.get("real_world_scale")),
        "academic_references_ge6": len(exp_data.get("references", [])) >= 6,
        "non_trivial_finding": bool(qg.get("non_trivial_finding")),
    }

    reasons: List[str] = []
    for k, ok in checks.items():
        if not ok:
            reasons.append(f"Missing criterion: {k}")

    n_ok = sum(1 for v in checks.values() if v)
    score = (n_ok / max(len(checks), 1)) * 10.0
    passed = all(checks.values())
    return QualityReport(
        passed=passed,
        score_10=score,
        checks=checks,
        reasons=reasons,
        recommended_action="eligible_for_zenodo_publish" if passed else "draft_only",
    )
