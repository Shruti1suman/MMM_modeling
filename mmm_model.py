# ===========================================
# Marketing Mix Modeling (MMM) - Script Version
# ===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import os

# ==============================
# STEP 1: Load Dataset
# ==============================
data_path = os.path.join("data", "MMM Weakly.csv")
data = pd.read_csv("MMM Weekly.csv")

print("\n✅ Data Loaded Successfully")
print("Columns:", list(data.columns))
print("Shape:", data.shape)

# ==============================
# STEP 2: Data Preparation
# ==============================
data['week'] = pd.to_datetime(data['week'])

spend_cols = [
    'facebook_spend', 'google_spend', 'tiktok_spend',
    'instagram_spend', 'snapchat_spend'
]

# Log-transform spends (log1p handles 0 safely)
for col in spend_cols:
    data[col + '_log'] = np.log1p(data[col])

# Features & target
features = [
    'facebook_spend_log', 'google_spend_log', 'tiktok_spend_log',
    'instagram_spend_log', 'snapchat_spend_log',
    'social_followers', 'average_price', 'promotions',
    'emails_send', 'sms_send'
]
X = data[features]
y = data['revenue']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# STEP 3: Model Training
# ==============================
tscv = TimeSeriesSplit(n_splits=5)
alphas = np.logspace(-3, 3, 50)
model = RidgeCV(alphas=alphas, cv=tscv).fit(X_scaled, y)

print("\n✅ Model Trained Successfully")
print("Best Alpha:", model.alpha_)

# ==============================
# STEP 4: Diagnostics
# ==============================
y_pred = model.predict(X_scaled)
residuals = y - y_pred

cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
print("CV R² Scores:", cv_scores)
print("Mean R²:", np.mean(cv_scores))

# Save residual plot
plt.figure(figsize=(10,5))
plt.plot(data['week'], residuals, label="Residuals")
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Analysis")
plt.legend()
plt.savefig("outputs/residuals.png")
plt.close()

# ==============================
# STEP 5: Insights
# ==============================
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:")
print(coef_df)

# Save coefficients plot
plt.figure(figsize=(10,6))
sns.barplot(data=coef_df, x="Coefficient", y="Feature")
plt.title("Feature Importance (Ridge Coefficients)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

print("\n✅ Results Saved in 'outputs/' folder")
