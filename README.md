# Online-Retail-E-Commerce-Customer-Data-Analysis

**Online Retail II** dataset (UK-based e-commerce transactions, Dec 2009 – Dec 2011), covering the full data science pipeline: cleaning, exploratory analysis, customer segmentation, churn prediction (classical ML and deep learning), and an integrated capstone.

## Project Structure

```
.
├── week1-data-cleaning/          # Data acquisition, cleaning, preprocessing
├── week2-eda/                    # Exploratory data analysis & visualization
├── week3-clustering/             # RFM feature engineering + K-Means segmentation
├── week4-supervised-learning/    # Churn classification (Logistic Regression, Random Forest)
├── week5-deep-learning/          # Churn classification (TensorFlow/Keras neural network)
└── week6-capstone/               # End-to-end pipeline + integrated recommendations
```

Each folder is self-contained: a notebook, a `data/` folder with the cleaned dataset, its own `requirements.txt`, and a short README. This top-level README gives the overall picture; open a week's folder for its full detail.

## Dataset

**Online Retail II** — invoice-level e-commerce transactions from a UK-based gift retailer, December 2009 to December 2011. 1,027,017 transaction line items, 4,932 products, 5,939 customers, 43 countries. Source: UCI Machine Learning Repository / Kaggle.    

**UCI Online Retail II - https://archive.ics.uci.edu/dataset/502/online+retail?utm_source=chatgpt.com**  
(use this link u get the dataset)

## Weekly Summary

### Week 1 — Data Acquisition, Cleaning & Preprocessing
Sourced the raw Online Retail II data, handled missing values (notably a missing Customer ID on 22.3% of rows), flagged cancelled orders (`IsCancellation`) rather than deleting them, and derived a clean `TotalAmount` field. Output: `online_retail_cleaned.csv`, used as the input for every subsequent week.

### Week 2 — Exploratory Data Analysis & Visualization
Tools: Pandas, Matplotlib, Seaborn. Summarized revenue trends, geographic and product concentration, and purchasing rhythm across 7 visualizations.
- Revenue is strongly seasonal, peaking every November ahead of Christmas.
- The UK accounts for ~85% of revenue.
- Invoice value and customer spend are both right-skewed (log-normal-ish).
- Purchasing concentrates on weekdays, 10:00–15:00.

### Week 3 — Unsupervised Learning & Clustering
Tools: Scikit-learn (K-Means, PCA). Engineered RFM (Recency, Frequency, Monetary) features per customer and clustered into 4 segments (k chosen via elbow method + silhouette score):

| Segment | Share | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|
| Champions | 14.9% | 38 days | 23.4 orders | £13,657 |
| Loyal Customers | 31.7% | 100 days | 5.9 orders | £2,076 |
| New / Occasional | 25.4% | 103 days | 1.8 orders | £461 |
| At Risk / Lapsed | 28.0% | 492 days | 1.7 orders | £521 |

### Week 4 — Supervised Learning
Tools: Scikit-learn (Logistic Regression, Random Forest). Predicted customer churn (no purchase after a 1-Sep-2011 cutoff) from 8 pre-cutoff behavioral features. Deliberately avoided leaking the `IsCancellation` flag (which is derived from negative quantities) into the feature set.

| Model | Accuracy | ROC-AUC | Recall |
|---|---|---|---|
| Logistic Regression | 73.9% | 0.807 | 78.5% |
| Random Forest | 74.8% | 0.813 | 75.7% |

### Week 5 — Deep Learning
Tools: TensorFlow/Keras. Re-solved the Week 4 churn problem with a small feedforward neural network (2 hidden layers, dropout + L2 regularization, early stopping) to benchmark against the classical models.

| Model | Accuracy | ROC-AUC | Recall |
|---|---|---|---|
| Neural Network | 73.6% | 0.803 | 82.3% |

**Finding:** all three model types (Weeks 4–5) converge to ~0.80–0.81 AUC — the engineered RFM-style features carry most of the available signal; added model complexity yields diminishing returns.

### Week 6 — Integrative Capstone
Combined every prior stage into one pipeline and cross-validated the Week 3 segments against the Week 4/5 churn labels on the same customers:

| Segment | Churn Rate |
|---|---|
| Champions | 8.6% |
| Loyal Customers | 34.6% |
| New / Occasional | 59.5% |
| At Risk / Lapsed | 100.0% |

The two independently-built analyses (unsupervised clustering, supervised classification) agree, validating both. Final recommendations: proactive retention for Champions, upsell campaigns for Loyal Customers, onboarding nudges for New/Occasional, and model-driven (not blanket) win-back targeting for At Risk/Lapsed customers.

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · TensorFlow/Keras · Jupyter

## How to Run

Each week's folder is independent:

```bash
cd week3-clustering
pip install -r requirements.txt
jupyter notebook Week3_Clustering_Analysis.ipynb
```

## Author

Virtual Data Science with Python Trainee Internship — 6-week program, Jul–Sep 2026.
