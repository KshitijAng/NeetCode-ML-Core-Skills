import torch
import torch.nn
from torchtyping import TensorType

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        M, N = to_reshape.shape
        reshaped = torch.reshape(to_reshape, (M * N // 2 , 2))
        return torch.round(reshaped,decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        averaged = torch.mean(to_avg, dim=0) # Computes the mean along the first dimension (rows)
        return torch.round(averaged, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        concatenate = torch.cat((cat_one,cat_two),dim=1) # Concatenates along dimension 1 (columns)
        return torch.round(concatenate,decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        loss = torch.nn.functional.mse_loss(prediction,target)
        return torch.round(loss,decimals=4)
