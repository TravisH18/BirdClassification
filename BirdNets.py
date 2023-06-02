import torch
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax, CrossEntropyLoss, BatchNorm2d, Dropout, Sequential, GELU

class BirdNetSimple(Module):
    def __init__(self):
        super(BirdNetSimple, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = Linear(128 * 28 * 28, 512)
        self.fc2 = Linear(512, 450)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class BirdNetComplexV1(Module):
    def __init__(self):
        super(BirdNetComplexV1, self).__init__()

        # layer 1
        self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm2d(16)
        self.relu1 = ReLU(inplace=True)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        # layer 2
        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(32)
        self.relu2 = ReLU(inplace=True)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # layer 3
        self.conv3 = Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = BatchNorm2d(64)
        self.relu3 = ReLU(inplace=True)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)

        # layer 4
        self.conv4 = Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = BatchNorm2d(128)
        self.relu4 = ReLU(inplace=True)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)

        # layer 5
        self.conv5 = Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = BatchNorm2d(256)
        self.relu5 = ReLU(inplace=True)
        self.pool5 = MaxPool2d(kernel_size=2, stride=2)

        # fully connected + dropout layers
        self.fc1 = Linear(256 * 7 * 7, 512)
        self.drop1 = Dropout(0.5)
        self.fc2 = Linear(512, 450)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = x.view(-1, 256 * 7 * 7)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu5(x)
        x = self.fc2(x)

        return x
    
class BirdNetComplexV2(Module):
    def __init__(self, dropout=0.5):
        super(BirdNetComplexV2, self).__init__()
        self.dropout = dropout
        # layer 1
        self.layer1 = Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )
        self.layer2 = Sequential(
            Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )
        
        self.layer3 = Sequential (
            Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )

        self.layer4 = Sequential (
            Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )

        self.layer5 = Sequential (
            Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )

        self.fc1 = Sequential(
            Linear(256 * 7 * 7, 1792, bias = True), #changed from 512 to 1792
            ReLU(inplace=True),
            Dropout(dropout)
        )
        self.fc2 = Linear(1792, 450, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class BirdNetSimpleV2(Module):
    def __init__(self):
        super(BirdNetSimpleV2, self).__init__()

        # step 1
        self.layer1 = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2)
        )

        #step 2
        self.layer2 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2)
        )

        # step 5
        self.layer3 = Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        
        #fc step
        self.fc1 = Sequential(
            Linear(128 * 28 * 28, 512, bias = True),
            ReLU(inplace=True),
            Dropout(0.5)
        )
        self.fc2 = Linear(512, 450)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 128 * 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class BirdNetComplexV3(Module):
    def __init__(self, dropout=0.5):
        super(BirdNetComplexV3, self).__init__()
        self.dropout = dropout
        # layer 1
        self.layer1 = Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(16),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.8, inplace=True)
        )
        self.layer2 = Sequential(
            Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(32),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )
        
        self.layer3 = Sequential (
            Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(64),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )

        self.layer4 = Sequential (
            Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(128),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )

        self.layer5 = Sequential (
            Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(256),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout, inplace=True)
        )

        self.fc1 = Sequential(
            Linear(256 * 7 * 7, 1792, bias = True), #changed from 512 to 1792
            ReLU(inplace=True),
            Dropout(self.dropout)
        )
        self.fc2 = Sequential(
            Linear(1792, 896, bias = True), #changed from 512 to 1792
            ReLU(inplace=True),
            Dropout(self.dropout)
        )
        self.fc3 = Linear(896, 450, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    