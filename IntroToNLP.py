import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, pos: List[str], neg: List[str]) -> TensorType[float]:
        # Create a Vocabulary
        words = set()
        combined = pos + neg
        for sentence in combined:
            for word in sentence.split():
                words.add(word)

        # Sort and Map Words to Integers
        sorted_list = sorted(list(words))
        word_to_int = {}
        for i, c in enumerate(sorted_list):
            word_to_int[c] = i + 1
        
        # Encode Sentences to Integer Sequences
        def encode(sentence):
            integers = []
            for word in sentence.split():
                integers.append(word_to_int[word])
            return integers

        # Encode All Sentences and Create Variable-Length Tensors
        var_len_tensors = []
        for sentence in combined:
            var_len_tensors.append(torch.tensor(encode(sentence)))

        # Pad Sequences for Uniform Length
        return nn.utils.rnn.pad_sequence(var_len_tensors, batch_first=True)
        
solution = Solution()
pos = ["Dogecoin to the moon"]
neg = ["I will short Tesla today"]
result = solution.get_dataset(pos,neg)
print(result)
