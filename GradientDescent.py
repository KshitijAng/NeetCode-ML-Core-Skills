# Your task is to minimize the function via Gradient Descent: f(x)=x**2

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: float) -> float:
        x = init  # Initial guess
        
        for _ in range(iterations):
            gradient = 2 * x  # Derivative of f(x) = x^2 is f'(x) = 2x
            x = x - learning_rate * gradient  # Gradient descent update step
        
        return round(x, 5)  # Round to 5 decimal places

# Example usage
solution = Solution()
result = solution.get_minimizer(iterations=0, learning_rate=0.01, init=5)
print(result)
