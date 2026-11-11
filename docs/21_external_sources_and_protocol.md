# External replication: source and baseline ledger

Access date: 2026-11-04.

The NanoDrone authors' repository documents real Crazyflie 2.1 Brushless flights
with synchronized motor inputs and motion-capture/state outputs, the
Square/Random/Chirp development trajectories and held-out Melon family:
https://github.com/idsia-robotics/nanodrone-sysid-benchmark

Archival paper: Riccardo Busetto, Elia Cereda, Marco Forgione, Gabriele Maroni,
Dario Piga, Daniele Palossi. Nonlinear system identification for a nano-drone
benchmark. Control Engineering Practice 172 (2026), 106871.
https://doi.org/10.1016/j.conengprac.2026.106871

Recent comparison context includes the benchmark's 2026 physics, neural,
hybrid-residual, and recurrent reference model families, and 2024 physics-guided
state-space augmentation (https://arxiv.org/abs/2405.10429). The extension tests
matched transition implementations to isolate context filtering. No original
reference-model leaderboard score is claimed, and no comparison against an
unexecuted method is entered as experimental evidence.

The original 2021--2026 literature corpus remains frozen. This dated external
source supplement extends the cutoff to 2026-11-04. The detailed pre-evaluation
design is in `plans/26_external_validity_revision.md`; download hashes, actual
sample counts, fitted parameters, and runtime are written by the experiment.

The authors' full preprint, Section 5.1 and Table 4, explicitly documents
offline timestamp synchronization, motor/acceleration lag alignment,
100 Hz resampling and fourth-order zero-phase low-pass filtering:
https://arxiv.org/html/2512.14450v1 . The published DOI/volume/author metadata
are independently verified; the accessible author manuscript establishes the
preprocessing details. Thus the external experiment evaluates publicly released,
preprocessed flight state estimates, not raw real-time sensor acquisition.

The pinned repository commit is
`2d921b57d166fe2debe08a5d39bd07297c5abc39`. No LICENSE/COPYING was found in
the inspected repository tree. The article's CC BY 4.0 license is not treated
as a license for the separate data/code repository. CSVs remain local research
downloads; reproduction uses the authors' URLs and LFS content hashes.
