from torch import nn

# Convolutional neural network (two convolutional layers)
class TemplateNet(nn.Module):
    def __init__(self, num_classes=1):
        super(TemplateNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #         self.fc = nn.Sequential(nn.Linear(2592, num_classes),
        #                        nn.ReLU())
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.5)
        self.fc = nn.Linear(113152, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dp1(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
