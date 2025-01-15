import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.logistic_regression import train_logistic_regression, predict
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pickle



file_path = './data/BC_data.csv'


df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()


X = df.drop(columns = ['diagnosis']) # Features
y = df['diagnosis'] # Target variable


# Normalizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print('Data Splitting Done!')


weights, bias = train_logistic_regression(X_train, y_train.values, lr=0.1, epochs=1000)

y_pred = predict(X_test, weights, bias)
accuracy = np.mean(y_pred == y_test.values)
print(f"Model Accuracy: {accuracy}")


conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{conf_matrix}")


report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
print(f"Classification Report: \n{report}")


model_params = {'weights': weights, "bias": bias}
with open('./models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model_params, f)

print("Model saved successfully")