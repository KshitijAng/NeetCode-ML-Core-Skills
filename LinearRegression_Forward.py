# Now that you've implemented get_model_prediction() for a model, it's time to implement the training loop.
import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, 
                       model_prediction: NDArray[np.float64], 
                       ground_truth: NDArray[np.float64], 
                       N: int, # note that N is just len(X)
                       X: NDArray[np.float64], 
                       desired_weight: int) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        weights = initial_weights.copy()
        N = len(X)  # Number of training samples

        # Gradient descent loop
        for _ in range(num_iterations):
            # Get model prediction
            model_prediction = self.get_model_prediction(X, weights)

            # Update each weight separately
            for i in range(len(weights)):
                # Compute the derivative for the i-th weight
                derivative = self.get_derivative(model_prediction, Y, N, X, i)

                # Update the i-th weight using gradient descent
                weights[i] -= self.learning_rate * derivative

        # Return the final weights rounded to 5 decimal places
        return np.round(weights, 5)

      
# Example Usage
solution = Solution()
X = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float64)
Y = np.array([6, 3], dtype=np.float64)
initial_weights = np.array([0.2, 0.1, 0.6], dtype=np.float64)

num_iterations = 10

result = solution.train_model(X,Y,num_iterations,initial_weights)
print(result)
