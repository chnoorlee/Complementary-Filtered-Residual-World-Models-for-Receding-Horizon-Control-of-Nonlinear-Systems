# Search and Baseline Contract v1

## Date window

Discovery and core comparison window: **2021-01-01 through 2026-08-05**. Older work may appear only as limited mathematical background, never as the sole or strongest empirical comparator.

## Search families

1. `world model reinforcement learning robust prediction`
2. `DreamerV3 world model uncertainty`
3. `TD-MPC2 latent dynamics SimNorm`
4. `IRIS transformer sample efficient world model`
5. `stochastic transformer world model`
6. `physics guided neural nonlinear dynamics`
7. `hybrid physics data driven dynamical systems`
8. `differentiable physics learned correction`
9. `uncertainty calibration dynamical systems`
10. `missing observations latent state model`
11. `long horizon rollout error world model`
12. `model based reinforcement learning survey 2023`

## Core comparator shortlist

| Comparator key | Recent source anchor | Protocol-matched implementation boundary | Why retained |
|---|---|---|---|
| RSSM-25 | Hafner et al., *Nature* 2025, DOI `10.1038/s41586-025-08744-2` | Recurrent stochastic state transition and Gaussian prediction head only; no actor, critic, imagination return, or image decoder | Current high-profile recurrent world-model family |
| SimNorm-WM-24 | Hansen et al., ICLR 2024, arXiv DOI `10.48550/arXiv.2310.16828` | Deterministic normalized latent transition under the shared supervised rollout loss; no MPC or RL objective | Recent scalable continuous-control world-model representation |
| TransWM-23 | Micheli et al., ICLR 2023, arXiv DOI `10.48550/arXiv.2209.00588` | Causal transformer transition on continuous state/action tokens; no Atari tokenizer or policy | Recent transformer world-model family |
| PGNN-22 | *Neural Networks* 2022, DOI `10.1016/j.neunet.2022.07.023` | Fixed physics-guided residual/fusion model using the same nominal simulator and matched capacity | Recent physics-guided comparator |
| PgDMM-22 | Liu et al., *Mechanical Systems and Signal Processing* 2022, DOI `10.1016/j.ymssp.2022.109276` | Probabilistic physics-guided latent transition adapted to observed state forecasting; no claim of exact unsupervised reproduction | Closest recent physics-plus-uncertainty comparator |

These names intentionally identify **transition-backbone adaptations**, not exact full-agent reproductions. Published task returns will not be copied into experimental tables.

## Fairness contract

- Same train/validation/test trajectories and normalization for every method.
- Same corrupted-context augmentation.
- Same tuning budget and early-stopping rule.
- Parameter counts within 20% where architecture permits; counts reported.
- All methods output predictive mean and variance and are evaluated with identical code.
- The physics-guided comparator and RGP-WM receive the same nominal simulator.
- All 15 prespecified cells are reported; no removal after seeing results.

## Verified source anchors already found

- DreamerV3 archival version: Hafner et al., “Mastering diverse control tasks through world models,” *Nature* (2025), DOI `10.1038/s41586-025-08744-2`.
- TD-MPC2: DBLP records ICLR 2024; preprint DOI `10.48550/arXiv.2310.16828`.
- IRIS: DBLP records ICLR 2023; preprint DOI `10.48550/arXiv.2209.00588`.
- Physics-guided nonlinear dynamics: *Neural Networks* (2022), DOI `10.1016/j.neunet.2022.07.023`.
- Scientific machine learning review: *Nature Reviews Physics* (2021), DOI `10.1038/s42254-021-00314-5`.
- Knowledge/physics integration survey: *ACM Computing Surveys* (2022), DOI `10.1145/3514228`.
- Dynamical systems physics-guided overview: *PNAS* (2024), DOI `10.1073/pnas.2311808121`.

## Gate risks

- Exact full-agent implementations are not protocol-compatible with low-dimensional supervised forecasting. The paper must say this explicitly and avoid a blanket SOTA claim.
- A materially equivalent reliability-supervised physics/data gate published in the date window would weaken novelty and trigger a method rename or pivot.
- Preprint-only sources may support context but not replace archival core evidence when an archival version exists.
- PRISM-WM (2025 preprint) uses a mode-discovery mixture of experts, HaM-World (2026 preprint) uses Hamiltonian residual structure and selective memory, and a 2026 equivariant-world-model preprint studies predictive-horizon certificates. They are required recency context but are not protocol-compatible core baselines: none supervises a physics-versus-data reliability gate in the present low-dimensional missing-observation protocol, and none had an archival version in the retrieved metadata snapshot.
