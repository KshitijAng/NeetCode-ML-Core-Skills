# Implement linear regression (without bias b).
import numpy as np
from numpy.typing import NDArray

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        pred = np.matmul(X,weights) # Matrix Multiplication
        return np.round(pred,5)


    def get_error(self, model_pred: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float: 
        error = np.mean(np.square(model_pred - ground_truth)) # Mean Sqaured Error (MSE)

        return np.round(error,5)
