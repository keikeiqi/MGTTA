import torch
import torch.nn as nn

import numpy as np

class MLPOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0, use_second_layer=False):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.Linear(4, hidden_sz)
        else:
            self.recurs = nn.Linear(2, hidden_sz)
        self.recurs2 = nn.Linear(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        self.use_second_layer = use_second_layer
        
    def forward(self, inp):
        if self.preproc:
            inp = inp.data
            inp2 = torch.zeros(inp.size()[0], 4, device=inp.data.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0:2][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 2:4][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
            inp2[:, 0:2][~keep_grads] = -1
            inp2[:, 2:4][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = inp2
        x = self.recurs(inp)
        if self.use_second_layer:
          x = self.recurs2(x)
        return torch.tanh(self.output(x))


class RNNOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0, use_second_layer=False):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(4, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1, bias=False)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        self.use_second_layer = use_second_layer
        
    def forward(self, inp, hidden, cell):
        if self.preproc:
            inp = inp.data
            inp2 = torch.zeros(inp.size()[0], 4, device=inp.data.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0:2][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 2:4][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
            inp2[:, 0:2][~keep_grads] = -1
            inp2[:, 2:4][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = inp2
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        if self.use_second_layer:
          hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
          return torch.tanh(self.output(hidden1)), (hidden0, hidden1), (cell0, cell1)
        else:
          return torch.tanh(self.output(hidden0)), (hidden0, ), (cell0, )


class DMOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0, use_second_layer=False):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.recurs = nn.LSTMCell(1, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1, bias=False)
        
    def forward(self, inp, hidden, cell):
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        
        return self.output(hidden0) * 0.01, (hidden0, ), (cell0, )
