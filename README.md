# Marketing Mix Modeling (MMM) - Weekly Dataset Analysis

## 📌 Overview
This repository contains a Marketing Mix Modeling (MMM) analysis using a 2-year weekly dataset.  
The goal is to **explain revenue as a function of marketing spend, promotions, pricing, and other drivers** while considering a causal mediation effect of Google spend (as a mediator between social/display channels and revenue).

The analysis is implemented in both:
- **`mmm_model.ipynb`** — interactive notebook with plots and insights  
- **`mmm_model.py`** — reproducible Python script for automated runs  

---

# ⚡ Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- jupyter

# ⚙️ Environment Setup

1. **Clone the repository**
```bash
git clone  https://github.com/Shruti1suman/MMM_modeling.git

```
2. Create virtual environment (if not already):
 ```bash
 python -m venv venv
venv\Scripts\activate 
```
3. Install dependencies
``` bash
pip install -r requirements.txt
```

#🚀 How to Run
 Jupyter Notebook
 ```
jupyter notebook mmm_model.ipynb
```
VS Code
```
python mmm_model.py
```

# 🧩 Dataset Columns
- week — Week of observation (datetime)
- facebook_spend, google_spend, tiktok_spend, instagram_spend, snapchat_spend — Paid media spend
- social_followers — Total social media followers
- average_price — Average product price
- promotions — Promotion spend/events
- emails_send, sms_send — Direct marketing channels
- revenue — Revenue (target variable)

# 🛠️ Modeling Approach
1. Data Preparation
- Converted week to datetime
- Log-transformed paid media spend (log1p) to handle zeros and scale distributions
- Standardized all features using StandardScaler
- Handled weekly seasonality implicitly using time-series CV

2. Causal Framing
- Assumed Google spend is a mediator between social/display channels and revenue
- Included log-transformed social media spend and Google spend as separate features
- Considered back-door paths to avoid leakage

3. Model
- Ridge Regression (with cross-validation)
- TimeSeriesSplit for validation (respecting temporal order)
- Hyperparameters (alpha) tuned using logarithmic grid search

4. Diagnostics
- Residual analysis to check model fit
- CV R² scores for stability
- Feature importance plots for interpretability
- Sensitivity to average_price and promotions explored

# 📊 Outputs
- All outputs (plots and tables) are saved in the outputs/ folder:
- residuals.png — Residual analysis over time
- feature_importance.png — Coefficients of each feature (interpretable importance)


# 🔍 Key Insights
- Social media spend drives Google search spend, which mediates revenue.
- Promotions and pricing have significant elasticity on revenue.
- Email/SMS campaigns show measurable but smaller effects.
- Ridge regression reduces multicollinearity and produces stable, interpretable coefficients.


# 📝 Author

-Shruti Suman
