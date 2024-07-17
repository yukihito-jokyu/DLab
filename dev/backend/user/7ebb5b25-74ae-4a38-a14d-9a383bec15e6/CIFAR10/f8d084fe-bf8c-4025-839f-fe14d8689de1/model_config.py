
import torch.nn as nn

class Simple_NN(nn.Module):
    def __init__(self):
        super(Simple_NN, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.1),
            nn.Flatten(),
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(64),
            nn.Linear(100, 10),

            )

    def forward(self, x):
        return self.seq(x)
