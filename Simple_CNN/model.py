import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_filters=64, dropout=0.3, n_layers=3):
        super(SimpleCNN, self).__init__()
        kennel_size = 3  # fixed kernel size
        layers = [
            nn.Conv2d(3, num_filters // 4, kennel_size, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(num_filters // 4, num_filters // 2, kennel_size, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        ]
        if n_layers == 3:
            layers += [nn.Conv2d(num_filters // 2, num_filters, kennel_size, padding=1), nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*layers)

        fc_input_dim = {
            2: (num_filters // 2) * 7 * 15,
            3: num_filters * 3 * 7
        }[n_layers]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
