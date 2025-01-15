from flask import render_template, request, redirect, url_for
from app import app
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'model' not in request.files or 'dataset' not in request.files:
        return redirect(request.url)
    
    model_file = request.files['model']
    dataset_file = request.files['dataset']
    
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_file.filename)
    
    model_file.save(model_path)
    dataset_file.save(dataset_path)
    
    return redirect(url_for('analyze', model_path=model_path, dataset_path=dataset_path))

@app.route('/analyze')
def analyze():
    model_path = request.args.get('model_path')
    dataset_path = request.args.get('dataset_path')
    
    model = joblib.load(model_path)
    data = pd.read_csv(dataset_path)
    
    X = data.drop('target', axis=1)
    y_true = data['target']
    
    y_pred = model.predict(X)
    
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png'))
    
    return render_template('results.html', report=report, cm_image='confusion_matrix.png')