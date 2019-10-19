from torch import nn
import torch.nn.functional as F

# 2 conv layers + 3 fully connected layers
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # layer 1, conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # layer 2, conv layer
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        # layer 3, fully connected layer
        self.fc1 = nn.Linear(16*5*5, 120)
        # layer 4, fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # layer 5, fully connected layer
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # pooling layer
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        # pooling layer
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet5()
print(net)
