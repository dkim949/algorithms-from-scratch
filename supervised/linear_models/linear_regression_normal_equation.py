import numpy as np
from sklearn.preprocessing import StandardScaler


class LinearRegressionNormalEquation:
    """
    Linear Regression using Normal Equation method.
    It scales the input features, adds an intercept term, and calculates the model parameters using the normal equation.
    """

    def __init__(self):
        # Initialize model parameters and scalers
        self.theta = None
        self.scaler_X = StandardScaler()

    def fit(self, X, y):
        # Fit the model using normal equation method

        # Scale the input feature
        X_scaled = self.scaler_X.fit_transform(X)

        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

        # Calculate theta using the normal equation
        # Normal equation: theta = (X^T * X)^-1 * X^T * y
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

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
