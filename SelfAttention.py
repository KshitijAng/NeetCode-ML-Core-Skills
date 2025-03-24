import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_gen = nn.Linear(embedding_dim, attention_dim,bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim,bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim,bias=False)    
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        scores = q @ torch.transpose(k,1,2)

        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores/ (attention_dim ** 0.5)

        lower_triangular = torch.tril(torch.ones(context_length,context_length))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask,float('-inf'))
        scores = nn.functional.softmax(scores,dim=2)

        return torch.round(scores @ v, decimals = 4)

solution = Solution(embedding_dim=3, attention_dim=4)
embedded=torch.randn(2, 2, 3)
print(solution)
output = solution(embedded)
print(output)
