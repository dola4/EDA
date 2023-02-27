# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

def eda(df, target_names, feature_names, target):
    # title ans markdown description
st.title("EDA and Predictive Model Dasboard")




    # add option 
    if st.checkbox("Show raw data"):
        st.write(df)

    # add option to show/hide missing value
    if st.checkbox("Show missing values "):
        st.write(df.isna().sum())
    if st.checkbox("Show data types"):
        st.write(df.dtypes)
    if st.checkbox("Show descriptive Statistics"):
        st.write(df.describe())
    if st.checkbox("Show correlation matrix"):
        corr = df.corr()
        mask = np.triu(np.ones_like(corr))
        sns.heatmap(corr, mask=mask, annot=True, cmap="coolware")
        st.pyplot()
    if st.checkbox("Show Histogram for each attributes"):
        for col in df.columns:
            fig, ax = plt.subplot()
            ax.hist(df[col], bins=20, density=9)
            ax.set_title(col)
            st.pyplot(fig)
    if st.checkbox("Show Density for each attributes"):
        for col in df.columns:
            fig, ax = plt.subplot()
            sns.kdeplot(df[col],fill=True)
            ax.set_title(col)
            st.pyplot(fig)
    if st.checkbox("Show Scatter plot"):
        fig = px.scaller(df, x=feature_names[0], y = feature_names[1])
        st.plotly_chart(fig)