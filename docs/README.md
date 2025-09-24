# ez-SMAD 🚀  
**Space Mission Analysis & Design Made Easy — Missile Tracking Edition**

---

## Overview
**ez-SMAD** is a modular, Python-based simulation framework for rapid evaluation of **space-based sensing architectures**, with a focus on **missile early warning and tracking**.  
It provides an end-to-end pipeline from **geometry & triangulation** to **filtering and performance assessment**, delivering reproducible results and publication-ready plots.

This release includes the first complete chain:
- Triangulation of targets from multi-satellite LOS vectors
- Export and reuse of **our own LOS geometry** (Earth-occlusion aware)
- Kalman Filter and EW-RLS estimators
- Automated parametric studies across LOS noise levels
- Rich diagnostic & performance plots
- PDF report builder for architecture assessment

---

## Features
- **Geometry**
  - Triangulation (`geometry/triangulate_icrf.py`) with analytic covariance, CEP50, and Monte Carlo validation
  - Automatic LOS export aligned to OEM epochs under `exports/geometry/los/`
  - Earth-occlusion gating of observers

- **Estimation**
  - **Kalman Filter (KF)** forward run (`estimationandfiltering/run_filter.py`)
  - **EW-RLS** tracks (`estimationandfiltering/ewrls_icrf_tracks.py`)
  - Metrics comparator (`comparator_new.py`) → `metrics.csv`

- **Studies & Orchestration**
  - Simple orchestrator (`orchestrator/main.py`) to run the full pipeline  
  - Parametric sweeps over σ_LOS (`orchestrator/parametric_study.py`)
  - Architecture performance reports (PDF) with plots and tuned “knobs”

- **Plots & Reports**
  - Error time series, CEP vs β-angle, scatter vs σ_LOS
  - RMSE component overlays (KF vs triangulation)
  - PDF reports summarizing performance for each run

---

## Directory Structure
