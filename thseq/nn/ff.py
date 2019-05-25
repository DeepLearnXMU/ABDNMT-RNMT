import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # y=w2(f(w1x))
        return self.linear2(self.dropout(self.relu(self.linear1(input))))

class Maxout(nn.Module):

    def __init__(self, input_size, hidden_size, pool_size):
        super().__init__()
        self.input_size, self.hidden_size, self.pool_size = input_size,hidden_size, pool_size
        self.lin = nn.Linear(input_size, hidden_size * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.hidden_size
        shape.append(self.pool_size)
        out = self.lin(inputs)
        m, i = out.view(*shape).max(-1)
        return m