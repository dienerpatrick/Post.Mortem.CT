import torch
from torch import nn
import torchvision
import torch.nn.functional as F

########################### Test Networks ###########################

class NeuralNetwork(nn.Module):
    """
    Default NN from pytorch tutorial https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

########################### ResNet18 ###########################

ResNet18 = torchvision.models.resnet18(pretrained=True)

for param in ResNet18.parameters():
    param.requires_grad = False

num_ftrs = ResNet18.fc.in_features
ResNet18.fc = nn.Sequential(
    nn.Linear(ResNet18.fc.in_features, 512), # in_features = 512
    nn.Dropout(0.5),
    nn.Linear(512, 3),
    nn.Dropout(0.5),
    # nn.Linear(10, 3),
    # nn.Dropout(),
    nn.Softmax()
)

########################### ResNet50 ###########################

ResNet50 = torchvision.models.resnet50(pretrained=True)

for param in ResNet50.parameters():
    param.requires_grad = False

num_ftrs = ResNet50.fc.in_features
ResNet50.fc = nn.Sequential(
    nn.Linear(ResNet50.fc.in_features, 128), # in_features = 512
    nn.Dropout(0.5),
    nn.Linear(128, 10),
    nn.Dropout(0.5),
    nn.Linear(10, 2)
)

########################### GoogleNet Inception V3 ###########################

InceptionV3 = torchvision.models.inception_v3(pretrained=True)

for param in InceptionV3.parameters():
    param.requires_grad = False

num_ftrs = InceptionV3.fc.in_features
InceptionV3.fc = nn.Sequential(
    nn.Linear(InceptionV3.fc.in_features, 512), # in_features = 2048
    nn.Dropout(0.3),
    nn.Linear(512, 10),
    nn.Linear(10, 2)
)


########################### AlexNet ###########################


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features= 9216, out_features= 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


AlexNetNP = AlexNet()


########################### AlexNetLight ###########################

# Alexnet without fully connected layer 3 (fc3)


class AlexNetLight1(nn.Module):
    def __init__(self):
        super(AlexNetLight1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features= 9216, out_features= 4096)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(in_features=4096 , out_features=2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


AlexNetLight1 = AlexNetLight1()


########################### AlexNetLight1 ###########################

# Alexnet with fc2 out_features=2

class AlexNetLight2(nn.Module):
    def __init__(self):
        super(AlexNetLight2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features= 9216, out_features= 4096)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(in_features= 4096, out_features=2)
        self.fc3 = nn.Linear(in_features=2 , out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


AlexNetLight2 = AlexNetLight2()


########################### AlexNetBN1 ###########################

# AlexNet with BatchNorm2D after conv1 and BatchNorm1D after fc1

class AlexNetBN1(nn.Module):
    def __init__(self):
        super(AlexNetBN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(96)
        self.batchnorm2 = nn.BatchNorm1d(4096)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features= 9216, out_features= 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=4)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


AlexNetBN1 = AlexNetBN1()

