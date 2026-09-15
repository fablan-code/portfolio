# portfolio
Portfolio - Python projects

# Data Science Portfolio — Facundo Blanco

Applied machine learning projects from my M.Sc. in Data Management and
Analytics (University of Buenos Aires, FCE). Each notebook documents the
problem, the approach, and what the results actually showed — including
the configurations that did not work.

---

## Hourly Electricity Demand Forecasting (LSTM)
`electricity_demand_lstm.ipynb` · Python · Keras · pandas

Forecasting French electricity demand on 50,397 hourly observations from
Open Power System Data (2015–2020), with linear interpolation of nulls,
min-max normalization, and sliding-window reshaping for sequence input.

Five controlled experiments varied window size (24h / 48h / 168h),
architecture depth, and the addition of hourly Paris temperature from the
Open-Meteo API. **The simplest configuration won**: a univariate 48-hour
window outperformed both the deeper network and the multivariate model,
suggesting seasonal and weather effects are already implicit in the demand
series itself. Extending the window to a full week degraded results
noticeably — more history meant more noise, not more signal.

Strictly chronological train/test split (2015–2018 / 2019–2020), no
shuffling. Evaluated on MAE and RMSE.

---

## Class Imbalance: SMOTE vs. a Tabular GAN
`class_imbalance_smote_vs_gan.ipynb` · Python · scikit-learn · imbalanced-learn · TensorFlow

How much does synthetic minority data actually help? Using the IBM HR
Employee Attrition dataset, a Random Forest classifier is trained under
three scenarios: original imbalanced data, data balanced with SMOTE, and
data balanced with a GAN trained exclusively on minority-class records.

Evaluated on precision, recall and F1 for the minority class — accuracy is
meaningless on an imbalanced target.

---

## Neural Networks vs. Tree Ensembles
`neural_nets_vs_tree_ensembles.ipynb` · Python · scikit-learn · Keras

Two supervised problems, one question: does a neural network beat gradient
boosting on mid-sized tabular data?

**Part 1 — Regression.** Predicting used-car prices, including feature
extraction from unstructured text fields (engine displacement, power,
torque), ordinal vs. one-hot encoding by variable type, and a log
transform of the target to correct skewness.

**Part 2 — Classification.** Credit default prediction on the UCI Credit
Card dataset (~22% default rate), using stratified splitting and balanced
class weights, evaluated on F1 and ROC-AUC. Both ensembles tuned with
RandomizedSearchCV over 3-fold cross-validation.

---

## Model Interpretability with SHAP
`model_interpretability_shap.ipynb` · Python · scikit-learn · SHAP

A deliberately parsimonious approach: logistic regression inside a
scikit-learn Pipeline (imputation, encoding, scaling), stratified 80/20
split, 5-fold cross-validation, and GridSearchCV over the regularization
parameter and solver.

Hyperparameter tuning improved mean cross-validated AUC but produced **no
improvement on the test set** — the baseline had already captured the
available structure. SHAP values were then used to decompose individual
predictions and verify that the model had learned relationships consistent
with the exploratory analysis, rather than artifacts.

---

Business Administration, UBA (cum laude) · M.Sc. candidate in Data
Management and Analytics · Previously PwC, Global Information Reporting
