import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("Titanic Survival Machine Learning App")
st.write("This app uses a Random Forest model to analyze passenger survival patterns.")

@st.cache_data
def load_and_train():
    train_df = pd.read_csv('train.csv')
    
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    train_df['IsAlone'] = np.where(train_df['FamilySize'] == 1, 1, 0)
    train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
    
    X = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], errors='ignore')
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    X['Embarked'] = X['Embarked'].fillna('S')
    
    X = pd.get_dummies(X, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    y = train_df['Survived']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, X

model, X_train = load_and_train()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Predict New Passenger Survival")
    
    p_class = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.slider("Age", 1, 80, 28)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)
    fare = st.slider("Ticket Fare ($)", 0, 500, 32)
    embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
    title = st.selectbox("Passenger Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

    input_data = pd.DataFrame([{
        'Pclass': p_class, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
        'FamilySize': sibsp + parch + 1, 'IsAlone': 1 if (sibsp + parch == 0) else 0,
        'Sex_male': 1 if sex == "Male" else 0,
        'Embarked_Q': 1 if "Q" in embarked else 0, 'Embarked_S': 1 if "S" in embarked else 0,
        'Title_Miss': 1 if title == "Miss" else 0, 'Title_Mr': 1 if title == "Mr" else 0,
        'Title_Mrs': 1 if title == "Mrs" else 0, 'Title_Rare': 1 if title == "Rare" else 0
    }])
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    if st.button("Calculate Survival Probability"):
        prob = model.predict_proba(input_data)[0][1]
        st.metric(label="Survival Chance", value=f"{prob:.1%}")
        if prob > 0.5:
            st.success("This passenger likely would have survived.")
        else:
            st.error("This passenger likely would not have survived.")

with col2:
    st.header("Model Insights: Feature Importance")
    st.write("Which characteristics matter the most to the algorithm's decisions?")
    
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    feat_importances.plot(kind='barh', color='#1f77b4', ax=ax)
    ax.set_title("Random Forest Feature Weightings")
    ax.set_xlabel("Relative Importance Score")
    plt.tight_layout()
    
    st.pyplot(fig)
