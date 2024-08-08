
import torch.nn as nn

class Simple_NN(nn.Module):
    def __init__(self):
        super(Simple_NN, self).__init__()

        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 10),

            )

    def forward(self, x):
        return self.seq(x)
