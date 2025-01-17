# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# from utils.helpers import load_model


# model_params = load_model('./models/logistic_model.pkl')
# weights = model_params['weights']
# bias = model_params['bias']


# file_path = './data/BC_data.csv'
# df = pd.read_csv(file_path)


# X = df.drop(columns=['diagnosis'])
# y = df['diagnosis']


# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


# linear_model = np.dot(X_scaled, weights) + bias
# y_pred_proba = sigmoid(linear_model)
# y_pred = (y_pred_proba > 0.5).astype(int)


# print("Confusion Matrix:")
# print(confusion_matrix(y, y_pred))


# print("\nClassification Report:")
# print(classification_report(y, y_pred))


# fpr, tpr, _ = roc_curve(y, y_pred_proba)
# roc_auc = auc(fpr, tpr)


# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Classifier')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()
# plt.savefig('confusion_matrix.png')
# plt.savefig('roc_curve.png')


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from models.logistic_regression import sigmoid, predict  # Import functions from your logistic_regression.py
import joblib

# Load the trained model
model_path = './models/logistic_model.pkl'
model = joblib.load(model_path)

weights = model['weights']
bias = model['bias']

# Load the dataset and preprocess
file_path = './data/BC_data.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Clean column names

X = df.drop(columns=['diagnosis'])  # Features
y = df['diagnosis']  # Target variable

# Normalizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load train-test split (as saved earlier in the main script)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluate on the test set
y_test_pred = predict(X_test, weights, bias)  # Use your custom `predict` function
accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred)

# Display results
print("Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
