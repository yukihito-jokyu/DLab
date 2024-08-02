
import torch.nn as nn

class Simple_NN(nn.Module):
    def __init__(self):
        super(Simple_NN, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(7, stride=1, padding=0),
            nn.Conv2d(128, 128, 7, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8192, 100),
            nn.ReLU(),
            nn.Linear(100, 10),

            )

    def forward(self, x):
        return self.seq(x)
