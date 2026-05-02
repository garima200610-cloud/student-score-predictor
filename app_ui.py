import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
st.write("Built using Machine Learning (Linear Regression)")
# Load data
data = pd.read_csv("data.csv")

# Train model
X = data[['Hours']]
y = data['Scores']
model = LinearRegression()
model.fit(X, y)

import matplotlib.pyplot as plt

# Plot graph
st.subheader("Study Hours vs Scores")
fig, ax = plt.subplots()
ax.scatter(data['Hours'], data['Scores'])
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Scores")
st.pyplot(fig)

# UI
st.title("Student Score Predictor")
st.write("Predict student score based on study hours (0–10 range)")

# Single input box
hours = st.number_input("Enter study hours:", min_value=0.0, max_value=10.0, step=0.5)

# Prediction
if st.button("Predict"):
    prediction = model.predict(pd.DataFrame([[hours]], columns=['Hours']))
    st.success(f"Predicted Score: {prediction[0]:.2f}")
