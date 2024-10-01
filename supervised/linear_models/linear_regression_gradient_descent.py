import numpy as np
from sklearn.preprocessing import StandardScaler


class LinearRegressionGradientDescent:
    """
    Linear Regression using Gradient Descent method.
    It scales the input features, adds an intercept term, and calculates the model parameters using gradient descent.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Initialize model parameters, hyperparameters, and scaler
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.scaler_X = StandardScaler()
        self.cost_history = []  # To store the cost at each iteration

    def fit(self, X, y):
        # Fit the model using gradient descent method

        # Scale the input features
        X_scaled = self.scaler_X.fit_transform(X)

        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

        # Initialize theta
        self.theta = np.random.randn(X_b.shape[1])

        # Gradient descent
        m = X_b.shape[0]  # number of samples
        for _ in range(self.n_iterations):
            # Compute predictions
            y_pred = X_b.dot(self.theta)

            # Compute error
            error = y_pred - y

            # Compute cost (MSE)
            cost = np.mean(error**2)
            self.cost_history.append(cost)

            # Compute gradient
            gradient = (2 / m) * X_b.T.dot(error)

            # Update theta
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        # Predict the target variable using the trained model
        X_scaled = self.scaler_X.transform(X)
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return X_b.dot(self.theta)

    def get_params(self):
        # Get the original scale parameters (slope and intercept)
        slope = self.theta[1:] / self.scaler_X.scale_
        intercept = self.theta[0] - np.sum(slope * self.scaler_X.mean_)
        return slope, intercept
