import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



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
