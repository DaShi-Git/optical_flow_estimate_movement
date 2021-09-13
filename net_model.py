import torch.nn as nn
import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
#         #self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
        
#         return x
class Net(nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2), #kernel_size
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 64, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 32, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32768, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, output_dim),
        )
        self.float()

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x