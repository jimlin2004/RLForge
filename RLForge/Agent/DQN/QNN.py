import torch as th

class QNN(th.nn.Module):
    def __init__(self, n_state: int, n_action: int, n_hidden: int):
        super().__init__()
        self.l1 = th.nn.Linear(n_state, n_hidden)
        self.relu1 = th.nn.ReLU()
        self.l2 = th.nn.Linear(n_hidden, n_hidden)
        self.relu2 = th.nn.ReLU()
        self.l3 = th.nn.Linear(n_hidden, n_action)
    def forward(self, state):
        out = self.l1(state)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out