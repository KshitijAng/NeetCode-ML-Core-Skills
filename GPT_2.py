# Make GPT Talk Back
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        res = []
        for i in range(new_chars):
            if len(context.T) > context_length:
                context = context[:, -context_length:]
            prediction = model(context) # B, T, Vocab_Size
            last_time_step = prediction[:, -1, :] # B, Vocab_Size
            probabilities = nn.functional.softmax(last_time_step, dim = -1)
            next_char = torch.multinomial(probabilities, 1, generator=generator)
            generator.set_state(initial_state)
            context = torch.cat((context, next_char), dim = -1)
            res.append(int_to_char[next_char.item()])
        return ''.join(res)
