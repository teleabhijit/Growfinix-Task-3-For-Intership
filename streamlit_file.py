import streamlit as st
import pandas as pd

st.title("CSV File Uploader Example")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read and display CSV
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("Here are the first few rows of your data:")
    st.dataframe(df.head())
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()
