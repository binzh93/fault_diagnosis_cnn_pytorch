import torch
import torch.nn.functional as F

class MyNet(torch.nn.Module):
    def __init__(self, num_calsses=4, init_weights=True):
        super(MyNet, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = torch.nn.Conv2d(3, 64, 3, 2, 1)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU(inplace=True)
        
        
        self.conv2_1 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_1   = torch.nn.BatchNorm2d(64)
        self.relu2_1 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_2   = torch.nn.BatchNorm2d(64)
        self.relu2_2 = torch.nn.ReLU(inplace=True)

     

        self.conv3_1 = torch.nn.Conv2d(64, 128, 1, 1)
        self.bn3_1   = torch.nn.BatchNorm2d(128)
        self.relu3_1 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3_2   = torch.nn.BatchNorm2d(128)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(128, 128, 1, 1)
        self.bn3_3   = torch.nn.BatchNorm2d(128)
        self.relu3_3 = torch.nn.ReLU(inplace=True)

        self.conv4_1 = torch.nn.Conv2d(128, 256, 1, 1)
        self.bn4_1   = torch.nn.BatchNorm2d(256)
        self.relu4_1 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4_2   = torch.nn.BatchNorm2d(256)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.conv4_3 = torch.nn.Conv2d(256, 256, 1, 1)
        self.bn4_3   = torch.nn.BatchNorm2d(256)
        self.relu4_3 = torch.nn.ReLU(inplace=True)
        
        self.last_conv = torch.nn.Conv2d(64+128+256, 512, 1, 1)
        self.last_bn   = torch.nn.BatchNorm2d(512)
        self.last_relu = torch.nn.ReLU(inplace=True)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.fc   = torch.nn.Linear(512, num_calsses)

        
        self.fc1   = torch.nn.Linear(512, 100)
        self.relu_ = torch.nn.ReLU(inplace=True)
        self.fc2   = torch.nn.Linear(100, num_calsses)




        
        if init_weights:
            self._initialize_weights()

    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        x_2 = x

        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        x = F.max_pool2d(x, 2)
        x_3 = x

        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        x = F.max_pool2d(x, 2)

        x_2 = F.max_pool2d(x_2, 4)
        x_3 = F.max_pool2d(x_3, 2)
        x = torch.cat((x_2, x_3, x), 1)
        
        x = self.last_relu(self.last_bn(self.last_conv(x)))
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
#         x = self.fc(x)
        x = self.relu_(self.fc1(x))
        x = self.fc2(x)
        

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)



