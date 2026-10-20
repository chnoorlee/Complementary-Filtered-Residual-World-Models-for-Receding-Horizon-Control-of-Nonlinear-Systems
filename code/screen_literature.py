"""Deterministic, auditable screening of the OpenAlex candidate pool."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RULES: list[tuple[str, str, str]] = [
    # World models and current transition backbones.
    (r"^Mastering diverse control tasks through world models$", "world_models", "archival DreamerV3 source"),
    (r"^TD-MPC2: Scalable, Robust World Models for Continuous Control$", "world_models", "ICLR 2024 baseline anchor"),
    (r"^Transformers are Sample-Efficient World Models$", "world_models", "ICLR 2023 transformer baseline anchor"),
    (r"^STORM: Efficient Stochastic Transformer based World Models", "world_models", "recent stochastic transformer world model"),
    (r"^GLAM: Global-Local Variation Awareness", "world_models", "AAAI 2025 state-space world model"),
    (r"^Prismatic World Model:", "world_models", "2025 preprint on mode-gated hybrid dynamics; novelty stress test"),
    (r"^HaM-World:", "world_models", "2026 soft-Hamiltonian residual world-model preprint; recency stress test"),
    (r"^(?:Certified World Models:|Scale Buys Interpolation, Structure Buys a Horizon:)", "world_models", "2026 predictability-certificate preprint"),
    (r"^DINO-WM:", "world_models", "recent offline feature-space world model"),
    (r"^UniZero:", "world_models", "recent scalable latent world model"),
    (r"^Bigger, Regularized, Optimistic:", "world_models", "recent scaled continuous-control comparator context"),
    (r"^QT-TDM:", "world_models", "archival transformer dynamics planning method"),
    (r"^Dream to Drive With Predictive Individual World Model$", "world_models", "archival applied world-model evidence"),
    (r"^Think2Drive:", "world_models", "recent long-horizon driving world model"),
    (r"^Curious Replay for Model-based Adaptation$", "world_models", "adaptation under changed dynamics"),
    (r"^HarmonyDream:", "world_models", "multi-task world-model context"),
    (r"^Model-based Reinforcement Learning: A Survey$", "world_models", "2023 field synthesis"),
    (r"^Understanding World or Predicting Future\? A Comprehensive Survey", "world_models", "2025 archival field synthesis"),
    (r"^A review of learning-based dynamics models for robotic manipulation$", "world_models", "2025 Science Robotics dynamics review"),
    (r"^Challenges of real-world reinforcement learning:", "world_models", "deployment-shift benchmark rationale"),
    (r"^Safe Learning in Robotics:", "world_models", "safety and reliability context"),
    # Hybrid physics and dynamical systems.
    (r"^Physics guided neural networks for modelling of non-linear dynamics$", "physics_hybrid", "recent fixed physics-guided baseline"),
    (r"^Physics-guided Deep Markov Models", "physics_hybrid", "closest probabilistic physics-guided baseline"),
    (r"^Physics-informed machine learning$", "physics_hybrid", "authoritative 2021 synthesis"),
    (r"^Integrating Scientific Knowledge with Machine Learning", "physics_hybrid", "authoritative hybrid-knowledge taxonomy"),
    (r"^Learning dynamical systems from data:", "physics_hybrid", "2024 dynamics-focused perspective"),
    (r"^Machine Learning With Data Assimilation and Uncertainty Quantification", "physics_hybrid", "dynamics and uncertainty synthesis"),
    (r"^Scientific Machine Learning Through Physics", "physics_hybrid", "2022 scientific-ML synthesis"),
    (r"^Hybrid physics-data-driven online modelling:", "physics_hybrid", "recent hybrid residual modeling"),
    (r"^Differentiable modelling to unify machine learning and physical models", "physics_hybrid", "differentiable hybrid-model perspective"),
    (r"^Learned turbulence modelling with differentiable fluid solvers:", "physics_hybrid", "multi-step physics-loss evidence"),
    (r"^Physics-Guided Machine Learning for Scientific Discovery:", "physics_hybrid", "physics-guided modeling evidence"),
    (r"^Respecting causality for training physics-informed neural networks$", "physics_hybrid", "recent training-horizon caution"),
    (r"^Perspectives on the integration between first-principles and data-driven modeling$", "physics_hybrid", "integration design context"),
    (r"^Hybrid Physics- and Data-Driven Dynamic State Estimation", "physics_hybrid", "partial-state hybrid estimation evidence"),
    (r"^Model fusion with physics-guided machine learning:", "physics_hybrid", "recent model-fusion comparison"),
    # Predictive uncertainty and calibration.
    (r"^A review of uncertainty quantification in deep learning:", "uncertainty", "authoritative recent UQ synthesis"),
    (r"^Uncertainty estimation with deep learning for rainfall", "uncertainty", "probabilistic sequence benchmark design"),
    (r"^Conformal prediction for uncertainty quantification in dynamic biological systems$", "uncertainty", "recent dynamic-system calibration method"),
    (r"^Accounting for model form uncertainty in Bayesian calibration of linear dynamic systems$", "uncertainty", "model-form uncertainty evidence"),
    (r"^Automatic calibration and uncertainty quantification in waves dynamical downscaling$", "uncertainty", "dynamic calibration evidence"),
]


def slug(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return "_".join(tokens[:5])


def quality_score(record: dict[str, Any]) -> float:
    score = 0.55
    if record.get("doi"):
        score += 0.10
    if record.get("abstract"):
        score += 0.05
    venue = (record.get("venue") or "").lower()
    if venue and "arxiv" not in venue and record.get("work_type") != "preprint":
        score += 0.20
    if (record.get("cited_by_count_snapshot") or 0) >= 10:
        score += 0.05
    return min(score, 1.0)


def main() -> None:
    input_path = Path("literature/candidates.jsonl")
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        title = record["title"].strip()
        for pattern, category, reason in RULES:
            if re.search(pattern, title, flags=re.IGNORECASE):
                # A preprint and its later archival version often have different
                # DOIs.  Deduplicate on normalized title, then keep the record
                # with the stronger venue/metadata score.
                key = record["normalized_title"]
                candidate = dict(record)
                candidate.update(
                    {
                        "category": category,
                        "screen_decision": "include",
                        "keep_reason": reason,
                        "relevance_score": 0.95 if category in {"world_models", "physics_hybrid"} else 0.85,
                        "quality_score": quality_score(record),
                        "cite_key": f"{(record.get('authors') or ['anon'])[0].split()[-1].lower()}{record.get('year')}_{slug(title)}",
                    }
                )
                existing = selected.get(key)
                if existing is None or candidate["quality_score"] > existing["quality_score"]:
                    selected[key] = candidate
                break

    shortlist = sorted(selected.values(), key=lambda item: (item["category"], -int(item["year"] or 0), item["title"]))
    output = {
        "screened_at": "2026-08-05",
        "date_window": ["2021-01-01", "2026-08-05"],
        "candidate_count": len(records),
        "included_count": len(shortlist),
        "screening_mode": "prespecified title-family rules followed by manual relevance audit",
        "shortlist": shortlist,
    }
    Path("literature/shortlist.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    with Path("literature/shortlist.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in shortlist:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for item in shortlist:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    print(f"candidate_count: {len(records)}")
    print(f"included_count: {len(shortlist)}")
    print(f"category_counts: {json.dumps(counts, sort_keys=True)}")


if __name__ == "__main__":
    main()
