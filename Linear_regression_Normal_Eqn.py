# Linear regression with Normal Equation
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

np.random.seed(1)
num_samples = 100

X = np.random.rand(num_samples)
Y = 5 * X + 0.3 * np.random.randn(num_samples)  # Noise added


class LinearRegression_NE:

    def __init__(self, x, y):
        #  Number of Samples
        self.samples = x.size
        # Augment 1 to every instance to accomodate for the Bias
        self.x = np.column_stack((np.ones_like(x), x))
        self.y = y

    # Compute Weight using the Normal Equations
    def train(self):
        XTX = np.dot(self.x.T, self.x) # X^T X
        XTY = np.dot(self.x.T, self.y) # X^T Y
        self.w = np.dot (np.linalg.inv(XTX), XTY)
        return self.w

    # Predict values for new data
    def predict(self, x):
        return np.dot(self.x, self.w)

model = LinearRegression_NE(X, Y)
weights = model.train()
print("w =", weights)

Y_HAT = model.predict(X)
fig= plt.figure(figsize=(10,6))
plt.title("Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Output")
plt.plot(X,Y, marker='o', linestyle='none')
plt.plot(X,Y_HAT)
plt.show()