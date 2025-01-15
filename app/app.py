from flask import Flask, request, jsonify
from utils.helpers import load_model
import numpy as np


app = Flask(__name__)


model_params = load_model()
weights = model_params['weights']
bias = model_params['bias']


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.jsonfeatures 
    features = np.array(data['features']).reshape(1, -1)
    linear_model = np.dot(features, weights) + bias
    probability = sigmoid(linear_model)
    prediction = int(probability > 0.5)
    return jsonify({'prediction': prediction, 'probability': probability.tolist()})


if __name__ == '__main__':
    app.run(debug=True)