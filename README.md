# 🏡 Ames Housing Price Prediction: Regression Diagnostics & Deployment

An end-to-end Machine Learning regression pipeline demonstrating feature selection, multicollinearity reduction (VIF), **LINE framework assumption validation**, feature engineering, and production deployment using **Streamlit**.

---

## 🚀 Live App & Repository Links

* **Live Interactive Streamlit App:** [Ames Housing Estimator App]([https://lnkd.in/e7CnStBR](https://ames-housing-regression-dkw9jkvvcnnmovw8zq7xp3.streamlit.app/)
* **GitHub Repository:** [ames-housing-regression](https://github.com/afadamarcello-code/ames-housing-regression)

---

## 💡 Key Takeaway & Business Impact

> **"More Features ≠ Better Model. Cleaner Features = Better Model."**

By diagnosing multicollinearity and applying domain-driven feature engineering, the feature space was reduced from **37 to 28 variables** while boosting predictive performance ($R^2$) from **0.8356 to 0.8999**—proving that simpler, well-validated statistical models outperform complex, noisy ones.

---

## 📊 Model Performance & Results

| Model Stage | $R^2$ Score | Feature Count | Workflow Description |
| :--- | :--- | :--- | :--- |
| **1. Baseline Model** | `0.8356` | 37 | Raw feature set without transformation |
| **2. Clean (Post-VIF)** | `0.8309` | 24 | Removed 13 redundant/multicollinear features |
| **3. Engineered + Log Target** | **`0.8999`** | **28** | **Best Model: Polynomials, Interactions & Log Target** |
| **4. Ridge Regularization** | `0.8999` | 28 | L2 Penalty applied ($\alpha$ optimized) |
| **5. Lasso Regularization** | `0.8995` | 28 | L1 Penalty applied |

---

## 📈 Visual Diagnostics & Analysis

### 1. Feature Selection & Multicollinearity
Analysis of feature correlations and Variance Inflation Factor (VIF) identified 13 redundant features.


---

### 2. Assumption Validation (LINE Framework)
Evaluating regression assumptions to guarantee unbiased linear estimates:
* **Lineating & Normality:** Applied logarithmic transformation to the target variable (`SalePrice`) to eliminate right-skewness and stabilize variance.
* **Homoscedasticity & Residuals:** Evaluated residual distribution post-transformation.

---

### 3. Feature Importance & Model Signals
Key predictive drivers identified by the final regression model:


---

## 🛠️ Data Pipeline & Methodology



[ Raw Features (37) ]
          │
          ▼ (Correlation Matrix + VIF Analysis)
[ Reduced Feature Set (24) ] ──► Removed 13 Redundant Signals
          │
          ▼ (Domain Feature Engineering)
[ Engineered Features (28) ] ──► OverallQual², GrLivArea², Qual×Area, HouseAge
          │
          ▼ (Target Transformation & LINE Validation)
[ Log Transformed Target ] ────► Fixed Skewness & Equal Variance
          │
          ▼ (Model Comparison & Regularization)
[ Final Model ($R^2 = 0.90$) ] ─► Exported to Scaler & Model Pickles (.pkl)
          │
          ▼ (Streamlit)
[ Web App / Production API ]


## 🧰 Tech Stack
Core Language: Python 3.x

Data Processing & Analysis: Pandas, NumPy

Machine Learning & Modeling: Scikit-Learn, Statsmodels

Visualization: Matplotlib, Seaborn

Deployment & Containerization: Streamlit





