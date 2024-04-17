import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
data = pd.read_csv('weatherAUS.csv')

# Data preprocessing
data.dropna(inplace=True)
le = LabelEncoder()
data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

# Drop irrelevant columns (like 'Date' and other non-numeric columns)
data.drop(['Date', 'Location'], axis=1, inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data)

# Define features and target variable
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logistic_pipeline = Pipeline([
    ('classifier', LogisticRegression())
])

# Naive Bayes Classifier
naive_bayes_pipeline = Pipeline([
    ('classifier', GaussianNB())
])

# Decision Trees
decision_tree_pipeline = Pipeline([
    ('classifier', DecisionTreeClassifier())
])

# Neural Network
neural_network_pipeline = Pipeline([
    ('classifier', MLPClassifier())
])

# Support Vector Machines (SVM)
svm_pipeline = Pipeline([
    ('classifier', SVC())
])

# Fit and evaluate models
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    print("ROC AUC Score:", auc_score)

# Evaluate models
models = {
    "Logistic Regression": logistic_pipeline,
    "Naive Bayes": naive_bayes_pipeline,
    "Decision Trees": decision_tree_pipeline,
    "Neural Network": neural_network_pipeline,
    "SVM": svm_pipeline,
}

print("Model Performance:")
for name, model in models.items():
    print("\n", name, "Performance:")
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

# Hyperparameter tuning
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # Regularization parameter for Logistic Regression
}

grid_search = GridSearchCV(logistic_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Check for overfitting/underfitting
print("\nModel Overfitting/Underfitting Check:")
for name, model in models.items():
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(name, "Training Accuracy:", train_score)
    print(name, "Testing Accuracy:", test_score)
