from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # layer 1, conv layer
        self.conv1 = nn.Conv2d(1, 6, 3)
        # layer 2, conv layer
        self.conv2 = nn.Conv2d(6, 16, 3)
        # layer 3, fully connected layer
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        # layer 4, fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # layer 5, fully connected layer
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # pooling layer
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # pooling layer
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
