# Ames Housing Price Prediction

## What This Project Demonstrates
End-to-end regression workflow: feature selection, multicollinearity handling,
assumption validation, feature engineering, and model deployment.

## Results
| Model | R² | Features |
|:---|:---|:---|
| Baseline | 0.8356 | 37 |
| Clean | 0.8309 | 24 |
| Engineered + Log | **0.8999** | 28 |
| Ridge | 0.8999 | 28 |
| Lasso | 0.8995 | 28 |

**Best model:** Linear Regression with log-transformed target + engineered features.

## Skills Shown
- Feature selection (correlation ranking + VIF)
- Multicollinearity diagnosis and removal
- Assumption validation (LINE framework)
- Feature engineering (polynomial, interaction, log transform)
- Model comparison and interpretation

## Key Insight
Started with 37 features, removed 13 due to redundancy, engineered 4 new ones,
and achieved **R²=0.90** — proving that cleaner, simpler models outperform
complex ones.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook ames_regression.ipynb