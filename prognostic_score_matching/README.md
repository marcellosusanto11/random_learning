# Prognostic Score Matching

Causal inference framework using **prognostic score matching** to evaluate the effect of decision variables on outcomes. Compares matched pairs with similar covariate-only predictions to isolate the incremental contribution of a treatment/decision variable.

## Overview

This notebook demonstrates prognostic score matching on two datasets:

### 1. Regression — House Price Prediction
- **Target:** `SalePrice`
- **Decision variable:** `OverallCond` (Overall Condition rating)
- **Covariates:** `LotArea`, `GrLivArea`, `GarageArea`, `TotalBsmtSF`, `YearBuilt`, `Neighborhood`, etc.
- **Metric:** MAE (Mean Absolute Error)
- **Result:** Including `OverallCond` improves MAE by ~9% (CatBoost) and ~6% (LGBM), with up to ~15% improvement when matched pairs have different decision values

### 2. Classification — Income Prediction (Adult Census)
- **Target:** `income_cat` (>50K vs <=50K)
- **Decision variable:** `age`
- **Covariates:** `relationship`, `marital.status`, `occupation`, `sex`, `hours.per.week`, `education.num`, `capital.gain`
- **Metric:** Log-loss
- **Result:** Including `age` improves log-loss by ~4-5%, with stronger gains in the same-decision subgroup (~13% for CatBoost)

## Method

1. Train two models per algorithm (CatBoost + LightGBM):
   - **Covariate-only model:** predicts outcome using covariates only
   - **Full model:** includes the decision variable with a monotone constraint
2. For each iteration:
   - Sample a random observation
   - Find a "match" with a similar covariate-only prediction (within threshold)
   - Record predictions from both models for both samples
3. Evaluate: compare MAE/log-loss of the full model vs covariate-only model across all matched pairs, stratified by same/different decision values

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

- catboost
- datasets (HuggingFace)
- lightgbm
- matplotlib
- numpy
- pandas
- scikit-learn
- scipy
- seaborn
- tqdm

## Project Structure

```
prognostic_score_matching/
├── prognostic_score_matching.ipynb   # Main analysis notebook
├── requirements.txt                  # Pinned dependencies
├── pyproject.toml                    # Ruff linting config
├── .gitignore
└── README.md
```

## Usage

```bash
source .venv/bin/activate
jupyter notebook prognostic_score_matching.ipynb
```
