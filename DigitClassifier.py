import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.l1 = nn.Linear(784,512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2) # 20% of the neurons will be randomly dropped (set to zero)
        self.l2 = nn.Linear(512,10) # final layer in a neural network
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        x = self.l1(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        out = self.sigmoid(x)
        return torch.round(out,decimals=4)
