# Marketing Mix Modeling (MMM)

## 📌 Overview
This repo contains an MMM analysis using weekly marketing spend and revenue data.

### Dataset Columns
- week
- facebook_spend, google_spend, tiktok_spend, instagram_spend, snapchat_spend
- social_followers, average_price, promotions
- emails_send, sms_send
- revenue

### Key Steps
1. Data preparation: log-transformed spends, scaling, handling zero-spend.
2. Modeling: Ridge regression with time-series cross-validation.
3. Diagnostics: Residual plots, CV R², stability checks.
4. Insights: Feature importance, risk discussion.

---

## 🚀 How to Run
```bash
git clone <this-repo>
cd MMM_Assignment
pip install -r requirements.txt
jupyter notebook mmm_model.ipynb
