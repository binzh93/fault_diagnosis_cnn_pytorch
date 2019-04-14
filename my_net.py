import torch
import torch.nn.functional as F

class MyNet(torch.nn.Module):
    def __init__(self, init_weights=True):
        super(MyNet, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1_1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)

        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)

        self.conv3_1 = torch.nn.Conv2d(128, 256, 3, 1, 1)
        self.relu3_1 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = torch.nn.ReLU(inplace=True)

        self.conv4_1 = torch.nn.Conv2d(256, 512, 3, 1, 1)
        self.relu4_1 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.conv4_3 = torch.nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = torch.nn.ReLU(inplace=True)
        
        self.conv5_1 = torch.nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_1 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        self.conv5_3 = torch.nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_3 = torch.nn.ReLU(inplace=True)

        self.fc1   = torch.nn.Linear(512*7*7, 1000)
        self.relu_fc1 = torch.nn.ReLU(inplace=True)
        self.fc2   = torch.nn.Linear(1000, 100)
        self.relu_fc2 = torch.nn.ReLU(inplace=True)
        self.fc3   = torch.nn.Linear(100, 4)
        
        if init_weights:
            self._initialize_weights()

    
    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = F.max_pool2d(x, 2)

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = F.max_pool2d(x, 2)
        
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = F.max_pool2d(x, 2)
        

        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = torch.nn.Dropout()(x)
        x = self.relu_fc2(self.fc2(x))
        x = torch.nn.Dropout()(x)
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

# MyNet()




