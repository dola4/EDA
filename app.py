# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import plotly.express as px
# import EDA
from eda import eda

#load data
dataset = load_iris()
#create dataframe
print(dataset)
data = dataset.data
feature_names = dataset.feature_names #colunms
target_names = dataset.target_names # classes
target = dataset.target # output
# create dataframe
df = pd.dataFrame(data, columns = target_names)
target = pd.Series(target)

#streamlit
# EDA
# setup app
st.set_page_config(page_title="EDA and ML Dasboard",
                   layout="centered",
                   initial_sidebar_state="auto")

# define sidebar option
options = ["EDA", "Predictive Modeling"]
selected_option = st.sidebar.selectbox("Selected Option", options)

if selected_option == "EDA":
    eda(df, target_names, feature_names, target)


#Predictive Modelling
elif selected_option == "Predictive Modeling":
    st.subheader("Predictive Modelling")
    st.write("Choose a transform type and Model")
    x = df.values
    y = target.values
    test_proportion = 0.30
    seed = 5
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_proportion, random_state=seed)
    transform_option = ["None", "StandardScaller", "Normalizer", "MinMaxScaler"]
    transform = st.selectedbox("Select data and transform")
    if transform == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif transform == "Normalizer":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif transform == "MinMaxScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    else:
        X_train = X_train
        X_test = X_test
    classifier_list = ["LogisticRegression", 
                   "SVM", 
                   "DecisionTree",
                   "KNeighbors",
                   "RandomForest"]
    classifier = st.selectbox("Select Classifier", classifier_list)
    # add option to select classifier
    # add logisticRegression
    if classifier == "LogisticRegression":
        st.write("Here are the result of a logistic regression")
        solver_value = st.selectedbox("Select solver",
                                      "lbfgs",
                                      "liblinear",
                                      "newton-cg",
                                      "newton-cb")
        model = LogisticRegression(solver)
        model.fix(X_train, Y_train)
        # make prediction
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average="micro")
        recall = recall_score(Y_test, Y_pred, accuracy)
        f1 = f1_score(Y_test, Y_pred, accuracy)
        st.write('Accuracy : {accuracy}' )
        st.write('precision : {precision}' )
        st.write('recall : {recall}' )
        st.write('f1 : {f1}' )
        st.write('confusion Matrix :' )
        st.write(cunfusion_matrix(Y_test, Y_pred))

    elif classifier == "DessisionTree":
        st.write("Here are the result of a dessision Tree")
        solver_value = st.selectedbox("Select solver",
                                      "lbfgs",
                                      "liblinear",
                                      "newton-cg",
                                      "newton-cb")
        model = LogisticRegression(solver)
        model.fix(X_train, Y_train)
        # make prediction
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average="micro")
        recall = recall_score(Y_test, Y_pred, accuracy)
        f1 = f1_score(Y_test, Y_pred, accuracy)
        st.write('Accuracy : {accuracy}' )
        st.write('precision : {precision}' )
        st.write('recall : {recall}' )
        st.write('f1 : {f1}' )
        st.write('confusion Matrix :' )
        st.write(confusion_matrix(Y_test, Y_pred))