import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up the Streamlit app
st.title('AI Trustworthiness Assessment Framework')

# File upload
model_file = st.file_uploader("Upload your model file", type=["pkl"])
dataset_file = st.file_uploader("Upload your dataset file", type=["csv"])

if model_file and dataset_file:
    # Save uploaded files to a temporary directory
    model_path = os.path.join("uploads", model_file.name)
    dataset_path = os.path.join("uploads", dataset_file.name)
    
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
    
    with open(dataset_path, "wb") as f:
        f.write(dataset_file.getbuffer())
    
    # Load the model and dataset
    model = joblib.load(model_path)
    data = pd.read_csv(dataset_path)
    
    X = data.drop('target', axis=1)
    y_true = data['target']
    
    # Perform predictions
    y_pred = model.predict(X)
    
    # Generate classification report and confusion matrix
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Display classification report
    st.subheader('Classification Report')
    st.json(report)
    
    # Plot and display confusion matrix
    st.subheader('Confusion Matrix')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)