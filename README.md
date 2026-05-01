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
