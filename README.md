# March Machine Learning Madness 

A complete, end-to-end machine learning pipeline built to predict the outcomes of the Men's and Women's NCAA Basketball Tournaments. This project was developed for the Kaggle March Machine Learning Madness competition and utilizes advanced feature engineering, rolling time-series metrics, and optimized XGBoost regressors to forecast exact matchup probabilities.

## 📊 Project Overview

The goal of this project is to accurately predict the probability of Team A beating Team B for every possible tournament matchup. Rather than treating this as a simple binary classification problem (Win/Loss), this pipeline predicts the **expected point differential** between two teams and converts that spread into a win probability using a tuned Gaussian Cumulative Distribution Function (CDF).

## 🧠 Model Architecture & Methodology

### 1. Data Pipeline & Feature Engineering
The models are trained on over a decade of historical NCAA regular-season and tournament data. To prevent data leakage, all features are calculated purely on a rolling, backward-looking basis.
* **Tempo-Free Advanced Stats:** Calculated Offensive and Defensive Efficiency Ratings (ORtg, DRtg) to normalize performance regardless of a team's pace of play.
* **Quadrant Records:** Grouped historical opponent strength into Q1, Q2, Q3, and Q4 buckets based on daily Massey Ordinals and Barttorvik rankings to track high-value wins.
* **Custom Elo Ratings (Women's Bracket):** Built a simulated Elo rating system from scratch to capture momentum and true team strength leading into the tournament.
* **Rolling Point Differentials:** Tracked cumulative averages and standard deviations of point differentials to measure both dominance and consistency.

### 2. The XGBoost Regressors
Two distinct `XGBRegressor` models were trained—one for the Men's bracket and one for the Women's bracket.
* Trained on a symmetric dataset (every game is duplicated so the model learns from both the winner's and loser's perspective).
* Hyperparameters optimized for early stopping (`eval_metric='mae'`) on a holdout validation season.
* Models are serialized and saved via `joblib` for rapid, memory-efficient deployment without needing to retrain.

### 3. Probability Conversion (The CDF Scale)
Predicting a 5-point win does not equal a 100% chance of winning. To convert the XGBoost point spread predictions into Kaggle-ready probabilities, the predictions are passed through a Normal CDF. The standard deviation (`scale`) of this distribution was mathematically optimized against the validation set's Brier Score:
* **Men's Optimal Scale:** `11.2`
* **Women's Optimal Scale:** `12.3`

## 📂 Repository Structure

The project has been modularized into specific notebooks for clean execution and reproducibility:

* **`Mens.ipynb`**: The data engineering, merging, and XGBoost training pipeline specifically tuned for the Men's dataset. Outputs `mens_xgboost_model.json`.
* **`Womens.ipynb`**: The custom Elo calculation, feature engineering, and XGBoost training pipeline for the Women's dataset. Outputs `womens_xgboost_model.json`.
* **`Submission_Merger.ipynb`**: A staging script that combines the predictions, bounds the limits to prevent infinite log-loss penalties on massive upsets, and formats the final `submission.csv` for Kaggle.
* **`Tournament_Oracle.ipynb`**: An interactive, localized querying tool that allows the user to input any two team names and instantly output the model's exact win probability and favored team.

## 🚀 Reproducibility

To run these notebooks locally, ensure you have the Kaggle March Madness datasets downloaded. 

**Requirements:**
\`\`\`text
pandas
numpy
xgboost
scikit-learn
scipy
joblib
matplotlib
\`\`\`

*(Note: Raw massive `.csv` and `.parquet` data files are strictly ignored via `.gitignore` to maintain repository health and comply with GitHub's file size limits. Ensure your local data folder matches the pathing in the notebooks).*
