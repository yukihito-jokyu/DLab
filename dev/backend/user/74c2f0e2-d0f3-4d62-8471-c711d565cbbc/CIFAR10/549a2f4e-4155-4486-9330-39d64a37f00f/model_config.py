
import torch.nn as nn

class Simple_NN(nn.Module):
    def __init__(self):
        super(Simple_NN, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(57600, 100),
            nn.ReLU(),
            nn.Linear(100, 10),

            )

    def forward(self, x):
        return self.seq(x)
