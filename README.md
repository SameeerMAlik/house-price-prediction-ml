# 🏠 House Price Prediction — Machine Learning Project

A complete, hands-on machine learning project that predicts house sale prices using the Kaggle **"House Prices: Advanced Regression Techniques"** dataset. Built  to learn ML fundamentals end-to-end.

---

## 📌 Project Overview

This project covers the full data science pipeline — from raw, messy real estate data to a clean, model-ready dataset with the most predictive features selected.

**Current progress:** EDA → Feature Engineering → Feature Selection ✅
**Next phase:** Model Building & Evaluation (in progress)

---

## 🔍 What This Project Covers

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, structure, and data type checks
- Missing value analysis (identifying columns with NaNs, calculating % missing per feature)
- Target variable (`SalePrice`) distribution analysis
- Correlation analysis between numerical features and `SalePrice`
- Temporal variable exploration (`YearBuilt`, `YrSold`, etc.)
- Discrete vs. Continuous variable identification
- Outlier detection

### 2. Feature Engineering
- **Missing value imputation**
  - Categorical: filled with `'Missing'` label
  - Numerical: filled with median (robust to outliers) + missing-value indicator flags
- **Temporal feature transformation** — converted raw years into "age at sale" (e.g., `YrSold - YearBuilt`)
- **Log transformation** — applied to skewed continuous features to normalize distribution
- **Rare category handling** — grouped infrequent categorical values (<1%) into a single `'Rare_var'` label
- **Categorical encoding** — Target-Guided Ordinal Encoding (categories ranked by mean `SalePrice`)
- **Feature scaling** — MinMaxScaler applied to normalize all features to a 0–1 range

### 3. Feature Selection
- Used **Lasso Regression** (`alpha=0.005`) with `SelectFromModel` to automatically identify the most predictive features
- Reduced ~82 engineered features down to the most impactful subset for modeling
- Final feature set used to build `X_train` for the model-building phase

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | Visualization |
| scikit-learn | Preprocessing, Lasso, feature selection |
| Jupyter Notebook | Development environment |

---

## 📁 Repository Structure

```
├── train.csv                  # Original Kaggle dataset
├── X_Train.csv                 # Cleaned, engineered, scaled dataset
├── house_price_prediction.ipynb  # Main notebook (EDA → Feature Engineering → Feature Selection)
├── README.md                   # Project documentation
└── .gitignore
```

---

## ⚠️ Known Limitations / Notes to Self

- Transformations in this version were applied to the **whole dataset before train/test split** (following the tutorial's teaching approach). In a production-safe pipeline, all statistics (median, scaling parameters, encoding maps) should be learned **only from training data** and applied to test data separately, to avoid data leakage. This will be corrected in a future iteration using `sklearn.Pipeline`.
- Feature selection threshold (`alpha=0.005`) can be tuned further via cross-validation for optimal results.

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction-ml.git
cd house-price-prediction-ml
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```

---

## 📚 Learning Source

This project was built while following **Krish Naik's Complete Machine Learning** playlist on YouTube, applying each concept (EDA, feature engineering, feature selection) hands-on using a real Kaggle dataset instead of just watching passively.

---

## 🔮 Next Steps

- [ ] Proper train/test split with leakage-safe pipeline
- [ ] Model building: Linear Regression, Lasso, Ridge, Random Forest, XGBoost
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Model evaluation (RMSE, R²)
- [ ] Basic deployment (Flask)
