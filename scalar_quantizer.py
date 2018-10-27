import torch
import torch.nn as nn
from torch.autograd import Variable

class quantize(nn.Module):
    def __init__(self, levels=[-1.0, 1.0], sigma=1.0):
        super(quantize, self).__init__()
        self.levels = levels
        self.sigma = sigma

    def forward(self, input):
        levels = input.data.new(self.levels)
        xsize = list(input.size())

        # Compute differentiable soft quantized version
        input = input.view(*(xsize + [1]))
        level_var = Variable(levels, requires_grad=False)
        dist = torch.pow(input-level_var, 2)
        output = torch.sum(level_var * nn.functional.softmax(-self.sigma*dist, dim=-1), dim=-1)

        # Compute hard quantization (invisible to autograd)
        _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        for _ in range(len(xsize)): levels.unsqueeze_(0)
        levels = levels.expand(*(xsize + [len(self.levels)]))

        quant = levels.gather(-1, symbols.long()).squeeze_(dim=-1)

        # Replace activations in soft variable with hard quantized version
        output.data = quant

        return output
