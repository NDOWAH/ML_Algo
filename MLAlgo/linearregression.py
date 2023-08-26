import numpy as np
class LinearRegression:

    def __int__(self, n_iteration = 1000, learning_rate = 0.001):
        self.n_iteration = n_iteration
        self.learning_rate = learning_rate
        self.weigt = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #initialized parameters
        self.weight = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iteration):
            y_predict = np.dot(self.weight, self.bias)
            dw = (1/n_samples) * np.dot(X.T, (y_predict - y))
            db = (1/n_samples) * np.sum(y_predict -y)
            # update the weight and the bias
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def prediction(self, X):
        predicted_y = np.dot(X, self.weight) + self.bias
        return predicted_y