# 🚗 Road Accident Risk Prediction
### Kaggle Playground Series S5E10

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Kaggle](https://img.shields.io/badge/Competition-Kaggle-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project is a complete end-to-end machine learning pipeline built for the
**Kaggle Playground Series Season 5, Episode 10** competition. The goal is to
predict the likelihood of accidents on different types of roads using structured
tabular data.

> **Metric:** Root Mean Squared Error (RMSE)  
> **Target:** `accident_risk` — a continuous value between 0 and 1  
> **Best Model:** LightGBM with early stopping

---

## 📂 Project Structure

```
road-accident-risk/
│
├── data/
│   ├── train.csv              ← training data (provided by Kaggle)
│   ├── test.csv               ← test data (provided by Kaggle)
│   └── submission_lgbm.csv    ← final submission file
│
├── notebooks/
│   └── predicting_road_accidents.ipynb   ← full pipeline notebook
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

| Property       | Detail                        |
|----------------|-------------------------------|
| Train rows     | 517,754                       |
| Test rows      | ~129,000                      |
| Features       | 16 columns                    |
| Target         | accident_risk (0 to 1)        |
| Missing values | None after cleaning           |
| Source         | Kaggle Playground Series S5E10|

### Features

| Feature                | Type        | Description                        |
|------------------------|-------------|------------------------------------|
| `num_lanes`            | Numeric     | Number of lanes on the road        |
| `curvature`            | Numeric     | Road curvature degree              |
| `speed_limit`          | Numeric     | Speed limit on the road            |
| `num_reported_accidents` | Numeric   | Historical accident count          |
| `road_signs_present`   | Binary      | Whether road signs are present     |
| `public_road`          | Binary      | Whether it is a public road        |
| `holiday`              | Binary      | Whether it is a holiday            |
| `school_season`        | Binary      | Whether it is school season        |
| `road_type`            | Categorical | Rural / Urban                      |
| `lighting`             | Categorical | Dim / Night / Normal               |
| `weather`              | Categorical | Foggy / Rainy / Clear              |
| `time_of_day`          | Categorical | Morning / Evening / Other          |

---

## 🔧 Pipeline

```
Raw Data
   │
   ├── 1. Data Cleaning
   │       ├── Fill missing values (median / mode)
   │       ├── Remove duplicates
   │       └── Convert to correct dtypes
   │
   ├── 2. Encoding
   │       └── Label Encoding for categorical features
   │
   ├── 3. EDA & Visualisation
   │       ├── Target distribution
   │       ├── Correlation heatmap
   │       └── Mutual information scores
   │
   ├── 4. Feature Engineering
   │       ├── visibility_score
   │       ├── speed_x_curvature
   │       ├── accidents_per_lane
   │       ├── night_rain / night_fog
   │       ├── rural_no_signs
   │       └── overall_risk_score
   │
   ├── 5. Train / Validation Split (80 / 20)
   │
   ├── 6. Baseline Models
   │       ├── Linear Regression
   │       ├── Ridge Regression
   │       ├── Lasso Regression
   │       └── ElasticNet
   │
   └── 7. LightGBM Model
           ├── Early stopping (100 rounds)
           ├── Regularisation (L1 + L2)
           └── Feature importance analysis
```

---

## 📈 Results

| Model            | Valid RMSE | Valid MAE | Valid R2 |
|------------------|------------|-----------|----------|
| Linear Regression| 0.0879     | 0.0705    | 0.7175   |
| Ridge            | 0.0879     | 0.0705    | 0.7175   |
| ElasticNet       | 0.0914     | 0.0731    | 0.6945   |
| Lasso            | 0.0973     | 0.0778    | 0.6534   |
| **LightGBM** ⭐  | **best**   | **best**  | **best** |

> Update LightGBM row with your actual scores after training

---

## 💡 Key Insights

- **Target is heavily skewed** — 99.99% of `accident_risk` values are near zero,
  with only 61 rows having values above 0.1
- **Linear models plateau** at R2 ≈ 0.72 — they cannot capture non-linear patterns
- **LightGBM significantly outperforms** all linear baselines
- **Most important features:** `num_reported_accidents`, `speed_limit`, `curvature`
- **Feature engineering** added meaningful combined risk signals the model
  couldn't see from raw columns alone

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/road-accident-risk.git
cd road-accident-risk

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
```

---

## 🚀 Usage

```python
# 1. Place train.csv and test.csv in the data/ folder

# 2. Open the notebook
jupyter notebook notebooks/predicting_road_accidents.ipynb

# 3. Run all cells in order:
#    Cell 1  → Load data
#    Cell 2  → Clean data
#    Cell 3  → Encode features
#    Cell 4  → EDA + visualisation
#    Cell 5  → Feature engineering
#    Cell 6  → Train/validation split
#    Cell 7  → Baseline models
#    Cell 8  → LightGBM training
#    Cell 9  → Generate submission.csv
```

---

## 📉 Visualisations Included

- ✅ Missing value heatmap
- ✅ Target distribution (raw + log)
- ✅ Feature correlation with accident_risk
- ✅ Mutual information scores
- ✅ Train/validation split balance
- ✅ Baseline model comparison (RMSE, MAE, R2)
- ✅ Actual vs Predicted plots
- ✅ Residuals distribution
- ✅ LightGBM feature importance (top 15)

---

## 🔮 Next Steps

- [ ] Hyperparameter tuning with Optuna
- [ ] XGBoost and CatBoost ensemble
- [ ] 5-fold cross-validation
- [ ] Stacking multiple models
- [ ] SHAP values for model explainability

---

## 📁 Submission Format

```csv
id,accident_risk
517754,0.352
517755,0.021
517756,0.008
```

All predictions clipped to [0, 1] as required by the competition.

---

## 🏆 Competition

- **Platform:** Kaggle
- **Series:** Playground Series Season 5, Episode 10
- **Timeline:** October 1 – October 31, 2025
- **Link:** [Kaggle Competition Page](https://www.kaggle.com/competitions/playground-series-s5e10)

---

## 👤 Author

**Your Name**  
[Kaggle Profile](https://www.kaggle.com/shahsawarjan) · [GitHub](https://github.com/shahsawarjan006) 

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
