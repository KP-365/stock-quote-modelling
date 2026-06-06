# Stock Quote Modelling with Conformal Prediction

Builds bid/ask quotes for a financial instrument by training regression models on 19,999 samples across 25 features, then applying conformal prediction to quantify uncertainty and set appropriate spreads.

## Problem

Given a dataset of stock observations, generate a point price prediction for an unseen sample and construct bid/ask quotes by estimating prediction uncertainty.

## Approach

1. Exploratory analysis: distribution plots, outlier detection, skewness (0.39) and kurtosis (7.0) of target
2. Model comparison across 7 regression algorithms
3. Leave-one-out cross-validation for residual analysis
4. Conformal prediction intervals used to derive spread width

## Models Compared

| Model | Type |
|---|---|
| Linear Regression | Linear |
| Ridge / Lasso / ElasticNet | Regularised linear |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| XGBoost | Gradient boosted trees |
| K-Nearest Neighbours | Non-parametric |

## Key Outputs

- Point predictions with uncertainty bounds for a held-out test sample
- Bid/ask spread derived from conformal prediction intervals
- Residual analysis across model families

## Libraries

`scikit-learn` `xgboost` `numpy` `pandas` `matplotlib` `scipy`

## How to Run

```bash
pip install scikit-learn xgboost numpy pandas matplotlib scipy
jupyter notebook stock_7_analysis.ipynb
```
