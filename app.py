# Import required libraries

import streamlit as st
import matplotlib.pyplot as plt

# Import trained objects
from training import (
    mse,
    mae,
    r2
)

from eda import df

# Page configuration
st.set_page_config(
    page_title="BigMart Sales Prediction",
    layout="centered"
)

# Page title
st.title("BigMart Sales Prediction")

# Display first 10 rows
st.subheader("Top 10 Rows of Dataset")

st.write(df.head(10))

st.write(
    f"{df.head(10).shape[0]} rows x {df.head(10).shape[1]} columns"
)

# Feature Selection Dropdown
st.subheader("Sales Comparison by Feature")

feature = st.selectbox(
    "Select Feature",
    [
        col for col in df.columns
        if col != 'Item_Outlet_Sales'
    ]
)

# Visualization
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df[feature],
    df['Item_Outlet_Sales']
)

ax.set_xlabel(feature)

ax.set_ylabel("Item Outlet Sales")

ax.set_title(
    f"Sales based on {feature}"
)

st.pyplot(fig)

# Model Evaluation
st.subheader("Model Evaluation")

st.write(
    f"Mean Squared Error: {mse:.2f}"
)

st.write(
    f"Mean Absolute Error: {mae:.2f}"
)

st.write(
    f"R² Score: {r2:.4f}"
)