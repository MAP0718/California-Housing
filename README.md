# California Housing Price Prediction

# Problem Statement
Predict the median house value in California districts using demographic and geographic features.
This is a supervised regression problem.


# Dataset
- California Housing dataset
- Contains numerical and categorical features
- Missing values present in numerical columns


# Preprocessing
- Numerical features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - One-hot encoding
- Implemented using `Pipeline` and `ColumnTransformer` to prevent data leakage.



# Models Used
- Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor (final model)



# Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Random Forest achieved lower error compared to baseline and tree models, indicating better capture of non-linear relationships.

---

# Cross-Validation
5-fold cross-validation was applied to the Random Forest model using RMSE as the evaluation metric.
This helps estimate generalization performance across multiple data splits instead of relying on a single evaluation.



# Key Learnings
- Importance of pipelines for clean ML workflows
- Baseline vs advanced model comparison
- Proper regression evaluation techniques
- Use of cross-validation for stability assessment

# Limitations & Future Work
- Hyperparameter tuning was not performed
- Further feature engineering could improve performance
- Model deployment can be added as an extension
