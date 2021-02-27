# IBM-HR-Analytics-Employee-Attrition-Performance


Dataset [link](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

Observations in the analysis are documented in medium [blog](https://binginagesh.medium.com/exploratory-data-analysis-of-ibm-hr-attrition-dataset-dae5f9e03e8e) and quora [blog](https://qr.ae/pNQHXP)

LinkedIn [post](https://www.linkedin.com/posts/activity-6771440072760922112-wQtI)

Machine Learning Algorithms are used to predict employee attrition but the results were not good.
Table below shows the mean F1 score of 10 fold cross validation 
| Algorithm | without SMOTE | with SMOTE |
| :---: | :---: | :---: |
| Linear regression | 41.57 | 48.32 |
| Support Vector MAchines | **50.06** | 41.28 |
| Decision Tree | 35.84 | 40.45 |
| Random Forest | 35.9 | 46.28 |
| XGBoost | 37.76 | 46.54 |
