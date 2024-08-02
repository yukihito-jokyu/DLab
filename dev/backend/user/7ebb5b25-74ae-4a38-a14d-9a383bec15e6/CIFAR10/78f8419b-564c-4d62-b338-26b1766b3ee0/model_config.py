
import torch.nn as nn

class Simple_NN(nn.Module):
    def __init__(self):
        super(Simple_NN, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(43264, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),

            )

    def forward(self, x):
        return self.seq(x)
