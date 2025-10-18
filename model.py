import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        self.use_shortcut =  stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out =self.conv1(x)
        out = self.bn1(out)
        out=torch.relu(out)
        out=self.conv2(out)
        out= self.bn2(out)
        shortcut=self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut
        out=torch.relu(out_add)
        return out