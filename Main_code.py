import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Title of the app
st.title('Cardiovascular Disease Prediction')

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Display dataset
    st.write("Dataset:")
    st.write(df.head())

    # Define features and target
    features = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
                'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory']
    target = 'HeartDisease'

    # Preprocess data
    X = df[features]
    y = df[target]
    
    # Convert categorical features to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def train_svm(X_train, y_train, X_test):
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    def train_random_forest(X_train, y_train, X_test):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    def train_decision_tree(X_train, y_train, X_test):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    def train_naive_bayes(X_train, y_train, X_test):
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    # Model selection
    model_name = st.selectbox('Select a model', ('SVM', 'Random Forest', 'Decision Tree', 'Naive Bayes'))

    if model_name == 'SVM':
        y_pred = train_svm(X_train, y_train, X_test)
    elif model_name == 'Random Forest':
        y_pred = train_random_forest(X_train, y_train, X_test)
    elif model_name == 'Decision Tree':
        y_pred = train_decision_tree(X_train, y_train, X_test)
    else:
        y_pred = train_naive_bayes(X_train, y_train, X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Display result
    st.write(f'Accuracy of {model_name}: {accuracy:.2f}')
