import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMNIST(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 1028)
        self.layer2 = nn.Linear(1028, 512)
        self.layer3 = nn.Linear(512, 64)
        self.layer4 = nn.Linear(64, 10)

    def forward(self, x_batch: torch.Tensor):

        _input = x_batch.view(-1, 784)
        out1 = F.relu(self.linear1(x_batch))
        out2 = F.relu(self.linear2(out1))
        out3 = F.relu(self.linear3(out2))
        out4 = F.relu(self.linear3(out3))

        return out4