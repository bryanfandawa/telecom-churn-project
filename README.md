# telecom-churn-project
This project builds a machine learning pipeline to predict customer churn for a telecommunications company. It covers data preprocessing, feature engineering, model training, model comparison, and interpretation of key factors that drive churn.

The main goals of this project are:
- Predict whether a customer will churn using machine learning models  
- Compare Logistic Regression, Random Forest, and XGBoost  
- Select the best-performing model based on ROC-AUC  
- Use Logistic Regression to interpret which features influence churn  

This project demonstrates an end-to-end ML process similar to what real companies use for customer retention analysis.

The dataset includes customer demographic information, subscription details, billing methods, services used, and whether they churned.  
Key columns include:
- Contract type
- Internet service type
- Monthly & total charges
- Payment method
- Extra services (online security, tech support, backup, etc.)
- Customer tenure
- Churn (target variable)


## Data Preprocessing & Modeling Steps
Steps performed in this project:

- Cleaned column names and handled missing values
- Cleaned and standardized variable formats using regular expressions
- Converted categorical variables into dummy/indicator variables
- Created interaction terms:
  - `paperless_x_month`
  - `tenure_x_month`
- Removed unnecessary or redundant columns
- Split the dataset into **train/test** sets
- Standardized numeric features (`tenure`, `monthly_charges`, `total_charges`, `paperless_x_month`, `tenure_x_month`)
- Applied **Recursive Feature Elimination (RFE)** to select the strongest predictors
- Performed cross-validation on multiple models to evaluate performance
- Tuned hyperparameters for Random Forest and XGBoost
- Compared model performance using **ROC-AUC** to select the final interpretation model


## Model Comparison
| Index | Model               | Accuracy | Recall | F1 Score | ROC-AUC |
|-------|---------------------|----------|--------|----------|---------|
| 2     | XGBoost             | 0.770    | 0.811  | 0.671    | 0.849   |
| 0     | Logistic Regression | 0.754    | 0.826  | 0.661    | 0.844   |
| 1     | Random Forest       | 0.778    | 0.743  | 0.660    | 0.844   |


## Coefficient Insights
Top features that **increase** churn odds:

| Feature | Odds Ratio | Meaning |
|--------|------------|---------|
| `contract_month-to-month` | **11.29×** | Month-to-month customers are far more likely to churn |
| `internet_service_fiber_optic` | **4.27×** | Fiber optic customers churn more than DSL |
| `contract_one_year` | **2.70×** | 1-year contracts churn more than 2-year contracts |
| `total_charges` | **1.48×** | Higher total spending increases churn |
| `payment_method_electronic_check` | **1.48×** | Strong churn indicator |

Features that **reduce** churn odds include:
- Online security
- Tech support
- Device protection
- Longer tenure


## Tools and Libraries
- Python
- Pandas, NumPy
- scikit-learn 
- XGBoost
- Matplotlib, Seaborn


## Conclusion
- XGBoost had the best predictive performance, but all models performed similarly.
- Logistic Regression was used to interpret feature effects due to its clear odds ratios.
- Contract type, internet service, payment method, and monthly/total charges are the strongest predictors of churn.
