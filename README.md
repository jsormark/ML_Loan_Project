# ML_Loan_Project
Project looked to provide a fictitious loan company with machine learning modeling recommendations to predict the probability a loan defaults for future clientele. The goal was to deliver to the client the best prediction model/s for identifying loan defaults utilizing data analysis and machine learning techniques.

# The project deliverables included:
- Identifying the best classifier to utilize for predicting a default on a loan.
- Providing a recommendation of the best model classifiers based on client presented business cases.
- Providing recommendations for risk assesment and gender focused business use cases.

# Strategy Overview - Objectives and Key Results
  - Objective: Improve the loan default prediction model/engine.
      - Key Result: Provide the recommendation for the best modeling classifier.
  - Objective: Provide scenario insights for loan default predictions.
      - Key Result: Identify the best modeling classifier for the business case scenarios.
  - Objective: Provide a classifier for predicting loan risk.
      - Key Result: Provide the recommended, the best performing based on project goal, classifier for predicting loan risk.
  - Objective: Validate gender based modeling performances.
      - Key Result: Present modeling data findings by gender and recommendations.

The resulting key benefits:
  - Reduction in total man hours to process defaulted loans.
  - Improves core competency, loan approval, provides value-add for clients and partners. 
      - Lead to other product initiatives, e.g. client risk calculator.
  - Provides better visibility of future projections and company roadmap.

# Dataset
Raw dataset source for the project was obtained from Kaggle.com. consisting of ~ 307,511 laon applications consisting of 122 attributes.
  - Data set was cleaned and pre-processed for analysis.

# Data Preparation
Data preparation involved the following,
  - 90% for Training and Validation, 10% for Testing
  - Out of the 90%, 80% was used for Training and 20% for Validation.
  - To address class imbalance and prevent model bias undersampling was performed.

# Model Evaluation
Model evalution utilized the following metrics to evaluate the model performance.
  - Accuracy: Measures overall correctness of loan default predictions.
  - Precision: Indicates how accurately the model predicts actual loan defaults.
  - Recall: Shows the model’s ability to identify all actual loan defaults.
  - F1-Score: Balances precision and recall, useful when loan default classes are imbalanced.
  - ROC AUC: Assesses the model’s ability to distinguish between default and non-default loans.

# Model Analysis
The project analyzed nine (9) different model classifiers for model optimization, as follows,
  - Decision Tree: Optimized max_depth with values [2, 6, 10].
  - k-Nearest Neighbors: Optimized n_neighbors with values [3, 4, 5, 6, 7, 8].
  - Logistic Regression: Default parameters with max_iter set to 1000.
  - Random Forest Classifier: GridSearchCV with cv=5. Parameters max_features values [2, 4, 6, 8, 10] and max_depth [6, 8, 10, 12, 14].
  - Support Vector Machine: GridSearchCV with cv=5. Parameters C and gamma with values np.logspace(-2, 2, 3).
  - Gradient Booster Classifier: GridSearchCV with cv=5. learning_rate with values from np.logspace(-2, 0, 3) and n_estimators with values [50, 100, 200] and max_depth of 6.
  - XG Booster Classifier: Default parameters.
  - Light GBM Classifier: Default parameters.
  - CatBoost Classifier: Default parameters.

# Modelling Results
Results were as follows,
  - Accuracy: LightGBM has the highest accuracy (0.6279), correctly predicting loan status 62.79% of the time.
  - Precision: LightGBM is the most precise (0.6309), correctly predicting a default payments 63.09% of the time.
  - Recall: CatBoost has the highest recall (0.6356), correctly identifying 63.56% of all default payments.
  - F1-Score: CatBoost and LightGBM have the highest F1-Scores (0.6317 and 0.6307), indicating a balance between precision and recall.
  - ROC AUC: LightGBM has the highest ROC AUC (0.6278), indicating the best overall performance across all classification thresholds.
  - Overall, LightGBM and CatBoost are the most effective models for predicting default payments

# Project Recommendations
Recommendations were as follows,
  - Risk Assessment: LightGBM - high accuracy and precision for correct loan status prediction.
  - Loan Approval: LightGBM - high precision for correct missed payment predictions.
  - Personalized Offers: LightGBM/CatBoost - high accuracy and precision for risk prediction, enabling personalized offers.
  - Early Intervention: CatBoost - high recall for identifying most actual missed payments, enabling proactive measures.
  - Automated Decision-Making: LightGBM/CatBoost - high accuracy and precision for speeding up the loan approval process.
  - Policy Development: LightGBM/CatBoost - high accuracy and precision for adjusting policies to reduce the risk of missed payments.

# Additional Business Cases
Specific business cases were analyzed. 

Business Case 1 - loan cases for the purpose of conducting repairs. Goal was to find the best model to avoid risky applicants specific to repair loans.
Model performance reuslts:
  - Accuracy: LightGBM has the highest accuracy (0.7), correctly predicting loan status 70% of the time.
  - Precision: LightGBM is the most precise (0.7279), correctly predicting default payments 72.79% of the time.
  - Recall: Decision Tree has the highest recall (0.7429), correctly identifying 74.29% of default payments.
  - F1-Score: LightGBM has the highest F1-Scores (0.7174), indicating a balance between precision and recall.
  - ROC AUC: LightGBM has the highest ROC AUC (0.6994), indicating the best overall performance across all classification thresholds.

Business Case 2 - Would it benefit to have a gender specific loan default model?
Model recommendation for this specific business case were:
  - Business Case 1, Risk Assessment: Use LightGBM (highest ROC AUC) for assessing loan risk as it considers both false positives, false negatives and the most effective models for predicting default payments.
  - Business Case 2, Gender Specific Loan Models: Best performance results were obtained when utilizing a combined dataset. Models showed that the only scenario for which to utlize a gender specific data set was when data was trained with male only data prediction of male defaults performed better.
