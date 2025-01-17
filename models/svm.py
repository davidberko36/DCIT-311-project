import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


file_path = '../data/BC_data.csv'
df = pd.read_csv(file_path)


df.columns = df.columns.str.strip()


X = df.drop(columns=['diagnosis'])
y = df['diagnosis']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm = SVC(random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}


grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)


print("Best Parameters: ", grid_search.best_params_)


best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


print("Tuned SVM Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)


model_path = './tuned_svm_model.pkl'
joblib.dump(best_svm, model_path)