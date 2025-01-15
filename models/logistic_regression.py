import numpy as np

def sigmoid(z):
    return 1 / (1 +np.exp(-z))


def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_logistic_regression(X_train, y_train, lr=0.01, epochs=1000):
    n_samples, n_features  = X_train.shape
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(epochs):
        linear_model = np.dot(X_train, weights) + bias
        y_pred = sigmoid(linear_model)

        dw = (1 /n_samples) * np.dot(X_train.T, (y_pred - y_train))
        db = (1 / n_samples) * np.sum(y_pred - y_train)

        weights -= lr * dw
        bias -= lr * db

        if epoch % 100 == 0:
            loss = compute_loss(y_train, y_pred)
            print(f'Epoch {epoch}: loss = {loss}')

    return weights, bias


def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_pred]