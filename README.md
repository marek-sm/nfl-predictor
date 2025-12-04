# 🏈 NFL Prediction System

A production-grade NFL prediction pipeline designed for:

- **Moneyline win probability**
- **Totals modeling (expected total + O/U probability)**
- Future expansion into spreads, props, live predictions, and alternate line distributions.

Built with **strict anti-leakage**, **chronological ordering**, and **modular architecture**.

---

# 🚀 Current Status (v0.3 — Steps 1–3 Complete)

## ✔ STEP 1 — Raw Data Loading

- Loader built on `nfl_data_py.import_schedules`
- Clean ingestion of:
  - scores, matchups, dates
  - `spread_line`, `total_line`, moneylines & odds
- Target creation (`home_win`, `total_points`)
- Full data quality suite:
  - duplicate detection
  - null checks
  - score validation
  - market sanity checks
  - chronological ordering
  - no “future completed games”

---

## ✔ STEP 2 — Base Dataset & Anti-Leakage Pipeline

- `build_base_dataset()`:
  - filters to completed games only
  - builds postseason/regular-season flags
  - global `game_index`
  - strict chronological sorting
- `load_base_dataset()` for future stages
- `split_by_season()`:
  - walk-forward time splits
  - **no future information leakage**
  - validated via time-based tests

---

## ✔ STEP 3 — Feature Engineering (Leak-Free)

### 🔧 Team-Level Features

- Points for/against, point differential
- ATS metrics
- Market-aware features
- **Elo ratings (pre-game)**
- Season-to-date:
  - `season_win_pct_to_date`
  - games played
- Rest/schedule:
  - `days_since_last_game`
  - `is_short_week`, `is_long_rest`, `coming_off_bye`

### 📈 Rolling Features (leak-free)

All rolling features use **groupby(team, season) + shift(1)**:

- Rolling mean/sum of points for/against
- Rolling point differential
- Rolling ATS margin
- Rolling total-vs-line
- Rolling win rate
- Rolling covered-spread rate

### 🏟 Game-Level Features

Each game is reconstructed with:

- `home_*` and `away_*` versions of all team stats
- Differential features like:
  - `diff_points_for_rolling_mean_*`
  - `diff_season_win_pct_to_date`
  - `diff_days_since_last_game`
  - `diff_implied_prob_ml`
  - **`diff_elo`**

### 🎯 Targets

- `target_home_win`
- `target_total_points`
- `target_total_over`

### 🧪 Test Suite Includes

- No-leakage tests (rolling + Elo)
- Home/away alignment
- Schedule rest correctness
- Market implied probability correctness
- Differential feature correctness
- End-to-end base dataset → features pipeline validation

✔ **24/24 tests passing — 85% coverage**

---

# 📂 Project Structure

```
nfl-predictor/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── src/nfl_predictor/
│   ├── data/
│   │   ├── loaders/
│   │   ├── preprocessing/
│   │   └── feature_engineering/
│   ├── evaluation/
│   ├── models/
│   ├── serving/
│   └── utils/
│
├── tests/
│   ├── test_games_loader.py
│   ├── test_data_quality.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│
├── run_data_check.py
├── run_base_dataset_check.py
├── run_feature_check.py
│
├── pyproject.toml
├── poetry.lock
└── README.md
```

---

# 🛠 Installation (Poetry Workflow)

## 1. Clone the repo

```bash
git clone https://github.com/marek-sm/nfl-predictor.git
cd nfl-predictor
```

## 2. Install dependencies with Poetry

```bash
poetry install
```

This will:

- Create/activate a project-specific virtual environment
- Install all dependencies from `pyproject.toml` and `poetry.lock`

## 3. Activate the virtual environment

Poetry 2.x:

```bash
poetry env activate
```

Verify:

```bash
python -c "import sys; print(sys.executable)"
```

---

# 🔍 Quick Checks

Run full sanity checks:

```bash
poetry run python run_data_check.py
poetry run python run_base_dataset_check.py
poetry run python run_feature_check.py
poetry run pytest -v
poetry run python -m compileall src
poetry run python -c "import nfl_predictor"
```

All tests should pass with no leakage and correct feature shaping.

---

# 🧠 Design Philosophy

- Zero leakage
- Deterministic, reproducible builds
- Test-driven development
- Modular, extensible architecture
- Sportsbook-aligned modeling (moneylines, totals, spreads)
- Production-grade engineering practices

---

# 🛣 Roadmap

## **Step 4 — Modeling**

- Moneyline (`XGBoostClassifier`)
- Totals (`XGBoostRegressor` + O/U classifier)
- Probability calibration (isotonic regression)

## **Step 5 — Weekly Predictions**

- Automatic market ingestion
- Feature generation
- Discord webhook outputs
- Model versioning & logging
- Confidence tiering + model agreement

## **Step 6 — Distribution Modeling**

- Residual bootstrapping
- PDFs / CDFs of totals
- Alternate line projections

---

# 👤 Author

**Marek Seablom-Michel**
