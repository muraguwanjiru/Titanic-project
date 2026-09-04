# Titanic Survival Predictor Web Application

An interactive Machine Learning web application that predicts passenger survival probability on the Titanic using an optimized Random Forest Classifier. Built completely in Python and deployed live to the cloud.

## Live Demo
You can access and interact with the deployed live web application here:
**https://muraguwanjiru-titanic-project-app-err3vy.streamlit.app/**

---

## What You Can Do With This Application

This web application translates an abstract machine learning model into an interactive sandbox. Visitors can use the interface to perform the following actions:

### 1. Simulate Custom Passenger Scenarios
Users can use the input form controls in the left sidebar to create hypothetical passengers or recreate real historical figures to see if they would have survived the disaster:
* **Toggle Demographics:** Alter the **Sex** and select historical **Social Titles** (like *Master* for young boys, *Mrs/Miss* for women, or *Rare* for royalty/officers) to see the dramatic impact of the "women and children first" protocol.
* **Manipulate Socioeconomic Factors:** Adjust the **Ticket Class (Pclass)** from 1st to 3rd class and slide the **Ticket Fare (\$)** wrapper to see how socioeconomic privilege directly altered survival probabilities.
* **Test Family Dynamics:** Input the exact number of **Siblings, Spouses, Parents, or Children** traveling together to observe how traveling alone versus traveling in a large family unit changes the safety outcome.
* **Adjust Age Distributions:** Drag the **Age** slider across a lifespan of 1 to 80 years old to identify exactly how age cutoffs affected a passenger's odds.

### 2. Compute Real-Time Survival Probabilities
* When a user modifies any input parameter and clicks the **"Calculate Survival Probability"** button, the underlying Random Forest model instantly formats, cleans, and structures those entries.
* The application runs the data through its 100 decision trees to generate a dynamic metric readout (e.g., *Survival Chance: 74.2%*).
* It provides clear visual feedback via color-coded alert flags: a **green notification banner** for predicted survival or a **red warning banner** if the passenger likely would have perished.

### 3. Analyze Model Insights & Algorithmic Logic
* Users can view the live, auto-updating **Feature Importance Graph** on the right side of the screen.
* This chart allows users to visually audit the inner mathematical mechanics of the machine learning model.
* It answers questions like: *Does the algorithm care more about ticket pricing (Fare) or passenger age?* (As shown in your graph, **Sex** and holding the title **"Mr"** carry the absolute highest analytical weight in the model's decision path).

---

## Tech Stack & Architecture
* **Language:** Python 3.10+
* **Data Processing & Engineering:** pandas, numpy
* **Machine Learning Framework:** scikit-learn
* **Data Visualization:** matplotlib, seaborn
* **Web UI Framework:** streamlit
* **Version Control & Hosting:** GitHub
* **Cloud Deployment Platform:** Streamlit Community Cloud

---

## Pipeline Overview

### 1. Advanced Feature Engineering
To maximize the predictive patterns extracted from the Kaggle dataset, the following feature manufacturing steps were performed:
* **Title Extraction:** Engineered a Title feature parsed via regex from the structural Name column to isolate social demographics (Mr, Mrs, Miss, Master, Rare).
* **Family Analytics:** Consolidated SibSp (siblings/spouses) and Parch (parents/children) into a holistic FamilySize tracking unit.
* **Isolation Mapping:** Formulated a binary indicator flag (IsAlone) marking passengers traveling entirely without accompanying relatives.

### 2. Robust Preprocessing
* **Imputation:** Missing data profiles across Age and Fare metrics were cleanly handled utilizing calculated median statistical parameters.
* **Vector Categorization:** Transformed categorical qualitative text descriptors (Sex, Embarked, Title) into algorithmic-ready numeric matrices using One-Hot Encoding (pd.get_dummies).
* **Dimensional Pruning:** Redundant identifier features (PassengerId, Ticket, Cabin) were cleanly scrubbed out.

### 3. Machine Learning Model Training
* **Core Algorithm:** RandomForestClassifier (Ensemble method aggregating 100 diverse decision trees).
* **Hyperparameters:** Configured with constrained architectural limitations (n_estimators=100, max_depth=5) to prevent overfitting.
* **Local Evaluation:** Achieved a **81.56%** local validation accuracy split.

---

## Project Repository Tree
```text
TITANIC_PROJECT/
├── app.py                # Main Streamlit web application engine
├── model.ipynb           # Local development Jupyter notebook pipeline
├── requirements.txt      # Production system dependencies list for Cloud deployment
├── train.csv             # Official Kaggle Titanic training feature set
├── test.csv              # Official Kaggle Titanic test validation target set
└── README.md             # Project documentation (This File)
```

---

## Local Setup & Execution Guide

To pull down this repository and host the application engine locally on your local environment, follow these execution steps:

### 1. Clone the Workspace
```bash
git clone https://github.com
cd Titanic-project
```

### 2. Environment Verification & Dependency Load
Install the required tracking environment blueprints matching our live cloud deployment:
```bash
pip install -r requirements.txt
```

### 3. Launch Local Streamlit Web Server
```bash
streamlit run app.py
```
*Your browser will automatically boot a preview window pointing towards http://localhost:8501*

This is work in progress
