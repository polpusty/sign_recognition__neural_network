import torch.nn


class Flatten(torch.nn.Module):
    def __init__(self, flatten_size):
        super(Flatten, self).__init__()
        self.flatten_size = flatten_size

    def forward(self, input_data, *args):
        return input_data.view(-1, self.flatten_size)
