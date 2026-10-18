# Changelog

## 2.0.0 — individual extension (Efe Akpınar)

Extends the team project beyond the report.

### Fixed
- Symptom-network scaling: a symptom without same-system partners had its transmissibility scaled to zero, so adding a harmless symptom (for example fever to cough) cancelled transmission entirely. Network scaling now amplifies transmissibility but never suppresses it. The report's parameter tables are reproduced unchanged.

### Added
- Infection-chain tracking (source and route of every infection) and measured reproduction numbers.
- Stochastic disease progression with variable stage durations and immunity-dependent progression to the severe stage.
- An optional latent period.
- Symptom-based self-isolation, an early-warning isolation trigger and a switch for the isolation phase.
- Parallel multi-seed ensembles and a Morris sensitivity screening of 13 settings the report leaves open.
- Calibration by approximate Bayesian computation against the 1978 English boarding-school influenza outbreak, with a deterministic SIR baseline and held-out checks.
- Planning experiments: interventions in the calibrated school model and a hospital-capacity sweep under two isolation policies.
- A Streamlit explorer, a reproducible analysis pipeline (`run_analyses.py`), four executed notebooks, JSON results and 34 tests.

## 1.0.0 — team project (team of 3)

Implementation of the symptom-based agent-based model described in the team's report by Ata Göktuğ Çim, Efe Akpınar and Ahmet Tuna Kuran. It includes:
- the symptom-network parameter engine;
- SIRD agents with immunity classes and health reserves;
- hospital triage, cafeteria contacts and the epidemic-phase isolation network;
- the report's three outbreak scenarios, with Kaplan–Meier survival analysis.
