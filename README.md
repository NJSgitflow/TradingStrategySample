# Quantitative Trading Strategy: ML-Driven Equity Panel 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository contains a comprehensive end-to-end machine learning pipeline for a **Long/Short Equity Strategy**. Built entirely within a single Jupyter Notebook, it demonstrates the full quantitative research workflow: from exploratory data analysis and rigorous statistical testing to rolling panel-data modeling, ensemble generation, and out-of-sample performance evaluation.

> ** Note on Data:** To ensure compliance and protect proprietary information, all datasets used in this public repository have been anonymized or replaced with **dummy data**. The focus of this repository is to demonstrate the architectural setup and quantitative methodology.

## Key Features & Workflow

1. **Data Exploration & Feature Engineering**
   - Conducted **PCA (Principal Component Analysis)** to assess dimensionality reduction and multicollinearity.
   - Applied **Augmented Dickey-Fuller (ADF) tests** to confirm stationarity and mean-reverting properties of the input features.
   - Generated technical features: 3D/7D/21D rolling returns, 21D rolling volatility, and 14D RSI based on relative market outperformance.
   - Applied **6-month rolling Z-scoring** to standardize features while strictly avoiding look-ahead bias.

2. **Rigorous Statistical Analysis (FDR)**
   - Addressed the Multiple Comparisons Problem across the generated features.
   - Applied the **Benjamini-Hochberg procedure** to control the False Discovery Rate (FDR).
   - *Result:* Ensured features contain true statistical signal rather than noise before feeding them into the models.

3. **Machine Learning Pipeline (Panel Data Approach)**
   - Leveraged a **multivariate Panel Data approach** across 30 industries, allowing models to learn non-linear cross-sectional effects and regime interactions.
   - Implemented a custom `run_rolling_model` function to train and predict on rolling windows.
   - Evaluated diverse algorithms: Linear/Ridge/Lasso Regressions, Logistic Regression, Random Forest, and Multi-Layer Perceptrons (MLP Regressor & Classifier).

4. **Model Evaluation & Ensembling**
   - Evaluated predictions using institutional quant metrics: **Information Coefficient (IC)**, **Rank IC (Spearman)**, and Hit Rate.
   - Constructed a robust **Meta-Ensemble** by standardizing and aggregating the predictions of the base learners to reduce variance.

5. **Backtesting & Monte Carlo Bootstrapping**
   - Simulated a continuous **Top-3 Long / Bottom-3 Short** portfolio strategy.
   - Calculated comprehensive risk/return metrics (Sharpe, Sortino, Calmar, Max Drawdown).
   - Validated the "Skill vs. Luck" factor using a **Bootstrapped Random Sampling ("Monkey Test")**, proving the Ensemble strategy ranks in the top percentiles of randomly generated portfolios out-of-sample.

---

## Repository Structure

```text
TradingStrategySample/
│
├── data/                               # Contains the anonymized dummy dataset (.csv)
├── Trading_Strategy_Pipeline.ipynb     # The main notebook containing the entire ML and Backtest workflow
├── requirements.txt                    # Python dependencies (pandas, scikit-learn, statsmodels, etc.)
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
