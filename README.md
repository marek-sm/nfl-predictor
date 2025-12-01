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
- Target creation:
  - `home_win`
  - `total_points`
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
  - validated via tests (time ordering, season isolation)

---

## ✔ STEP 3 — Feature Engineering (Leak-Free)

Production-grade team & game-level features including:

### 🔧 Team-Long Features (one row per team per game)

- Points for/against, point differential
- ATS metrics (`ats_margin`, `covered_spread`)
- Market-aware features (`implied_prob_ml`, `total_vs_line`)
- Season-to-date stats:
  - `season_win_pct_to_date` (shifted expanding mean)
- Schedule/rest features:
  - `days_since_last_game`
  - `games_played_season_to_date`
  - `is_short_week`, `is_long_rest`, `coming_off_bye`

### 📈 Rolling Features (leak-free)

Grouped by `["team", "season"]` with `shift(1)`:

- `points_for_rolling_mean/sum_{3,5,8}`
- `points_against_rolling_mean/sum_{3,5,8}`
- `point_diff_rolling_mean_*`
- `ats_margin_rolling_mean_*`
- `total_vs_line_rolling_mean_*`
- `team_win_rate_rolling_mean_*`
- `covered_spread_rate_rolling_mean_*`

### 🏟 Game-Level Features

Reconstructed into **one row per game**:

- `home_*` and `away_*` versions of all team-level features
- Matchup differential features:
  - `diff_points_for_rolling_mean_*`
  - `diff_point_diff_rolling_mean_*`
  - `diff_season_win_pct_to_date`
  - `diff_days_since_last_game`
  - `diff_implied_prob_ml`
  - and more…

### 🎯 Targets

- `target_home_win`
- `target_total_points`
- `target_total_over`

### 🧪 Test Suite Includes

- rolling no-leakage tests
- timing correctness
- home/away alignment
- schedule & season aggregate correctness
- implied probability correctness
- diff feature correctness
- end-to-end integration (real `nfl_data_py` schedule data)

👉 **23/23 tests passing.**

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
├── requirements.txt
└── README.md
```

---

# 🛠 Installation

### 1. Clone the repo

```bash
git clone https://github.com/marek-sm/nfl-predictor.git
cd nfl-predictor
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

---

# 🔍 Quick Checks

```bash
python run_data_check.py
python run_base_dataset_check.py
python run_feature_check.py
pytest -v
```

---

# 🧠 Design Philosophy

- Zero leakage
- Test-driven development
- Modular, extensible architecture
- Realistic sportsbook pipeline (moneylines, spreads, totals)
- Production-grade code structure

---

# 🛣 Roadmap

### **Step 4 — Modeling**

- Moneyline (XGBoostClassifier)
- Totals (XGBoostRegressor + Classifier)
- Calibration (isotonic)

### **Step 5 — Weekly Predictions**

- Automated market ingestion
- Feature generation
- Discord webhook output
- Confidence tiering
- Model versioning

### **Step 6 — Distribution Modeling**

- Residual bootstrapping
- PDF/CDF of totals
- Alternate line projections

---

# 👤 Author

**Marek Seablom-Michel**
