import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embeddings = self.embedding_layer(x)
        averaged = torch.mean(embeddings, axis = 1)
        projected = self.linear_layer(averaged)
        return torch.round(self.sigmoid_layer(projected), decimals=4)

model = Solution(vocabulary_size=5)
print(model)
x = torch.tensor([[4, 1], [2, 3]])
output = model(x)
print(output)
