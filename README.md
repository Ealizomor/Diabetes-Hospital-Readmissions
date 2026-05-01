# Predicting 30-Day Hospital Readmissions in Diabetic Patients Using Machine Learning

# Business Problem / Motivation

Hospital readmissions within 30 days are a major challenge in healthcare systems, leading to increased costs, reduced quality of care, and potential financial penalties for hospitals. For patients with chronic conditions such as diabetes, the risk of readmission is particularly high due to ongoing disease management and complications.

The key problem is that healthcare providers often lack accurate, data-driven tools to identify which patients are at high risk of being readmitted after discharge.

This project aims to address that gap by developing a predictive model that can identify high-risk patients early. From a clinical and business perspective, the priority is to minimize false negatives, since failing to identify a high-risk patient can result in missed interventions, worsening health outcomes, and higher costs.

By improving early risk detection, hospitals can implement targeted interventions such as follow-up care, medication management, and patient monitoring.


# Project Overview

This project develops a machine learning model to predict 30-day hospital readmissions using the UCI Diabetes dataset, which contains over 100,000 hospital encounters.

A complete data science pipeline was implemented, including data preprocessing, feature engineering, handling class imbalance, and iterative model development. Multiple models were evaluated, including Decision Trees, Random Forest, and boosting methods.

The final model, LightGBM, was selected for its ability to handle high-cardinality clinical features and capture nonlinear relationships.

- Final Recall: ~64–65%
- Baseline Recall: 22% (Decision Tree)

This represents a substantial improvement in identifying high-risk patients. Additionally, SHAP was used to provide model interpretability, ensuring that predictions are transparent and aligned with clinical reasoning.

Overall, this project demonstrates how machine learning can be applied to support data-driven clinical decision-making and reduce preventable hospital readmissions.


# Data
- Source: UCI Machine Learning Repository – Diabetes 130-US Hospitals Dataset
  https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Type: Structured healthcare data (tabular)
- Size:
  - ~102,000 patient encounters
  - 47 features
- Key Features:
  - Demographics: age, race, gender
  - Clinical Information: diagnosis codes (primary, secondary), lab results (A1C, glucose)
  - Medications: insulin, diabetes drugs, medication changes
  - Healthcare Utilization: number of emergency visits, inpatient stays, outpatient visits
  - Hospital Information: admission type, discharge disposition, payer code

This dataset provides a comprehensive view of patient encounters, combining clinical, behavioral, and administrative factors relevant to readmission risk.

<img width="1028" height="667" alt="image" src="https://github.com/user-attachments/assets/7fce022b-f789-42cf-8fb9-5ed558da7d2e" /> 

Insight:

- The dataset is highly imbalanced:
  - Majority class: No readmission
  - Minority class: <30 days readmission
    
- This imbalance explains why early models struggled with recall.
- It also justifies:
  - Combining “<30” and “>30” into one class
  - Using recall as the primary evaluation metric
  - Testing sampling strategies like SMOTE

<img width="853" height="737" alt="image" src="https://github.com/user-attachments/assets/b5f03d6d-5ed0-4040-91e1-a048ac5f88e5" />

Insight:

- Moderate correlation between:
  - time_in_hospital ↔ num_medications (0.47)
  - num_medications ↔ num_procedures (0.39)
    
- Indicates that patients with longer stays tend to receive more treatments.
- Most features show low to moderate correlation, meaning:
  - Low multicollinearity
  - Features provide independent predictive value
 
## Key Takeaways from EDA
  - Strong class imbalance required special handling
  - Healthcare utilization features are closely related
  - Relationships are not purely linear, supporting use of ensemble models
  - Feature independence suggests models like boosting can perform well

# Modeling Approach
 - ## Baseline Model
   - Decision Tree
   - Recall: 22%
   - Highlighted severe bias toward majority class
   
<img width="801" height="388" alt="image" src="https://github.com/user-attachments/assets/a9a23173-af12-4c72-b7e3-4918a161d0bc" />

- ## Advanced Models
  - Random Forest
  - Decision Tree
  - Ensemble Modeling
  - LightGBM (Final Model)
    
 - ## Why These Models?
   Random Forest - Improved upon Decision Trees by combining multiple trees to reduce overfitting and improve generalization. It is effective for structured data and can capture more complex patterns than a single tree.

   Ensemble Modeling (Boosting) - Improved performance by focusing on errors from previous models, making them well-suited for difficult problems like imbalanced classification.

   LightGBM - Achieved approximately 64–65% recall, significantly improving the model’s ability to detect high-risk patients compared to the baseline.
   
