from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import os

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train models
models = {
    'logistic_regression': LogisticRegression(max_iter=200),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'svm': SVC(probability=True)
}

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'models/{name}_model.pkl')

print("Models trained and saved successfully.")