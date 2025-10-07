import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool + double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv - using bilinear upsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample the deeper feature map
        x1 = self.up(x1)
        
        # Handle padding if sizes don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channels dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetpp(nn.Module):
    """Correct UNet++ implementation with all dense skip connections"""
    def __init__(self, n_channels=1, n_classes=1, base_filters=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder path
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4) 
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        # Bridge
        self.bridge = Down(base_filters * 8, base_filters * 16)
        
        # ALL the upsampling nodes for UNet++ (dense connections)
        # Level 1 nodes (first row after encoder)
        self.up_01 = Up(base_filters * 2 + base_filters, base_filters)
        
        # Level 2 nodes (second row)
        self.up_02 = Up(base_filters * 2 + base_filters * 2, base_filters)
        self.up_12 = Up(base_filters * 4 + base_filters * 2, base_filters * 2)
        
        # Level 3 nodes (third row)  
        self.up_03 = Up(base_filters * 2 + base_filters * 3, base_filters)
        self.up_13 = Up(base_filters * 4 + base_filters * 2 * 2, base_filters * 2)
        self.up_23 = Up(base_filters * 8 + base_filters * 4, base_filters * 4)
        
        # Level 4 nodes (fourth row)
        self.up_04 = Up(base_filters * 2 + base_filters * 4, base_filters)
        self.up_14 = Up(base_filters * 4 + base_filters * 2 * 3, base_filters * 2)
        self.up_24 = Up(base_filters * 8 + base_filters * 4 * 2, base_filters * 4)
        self.up_34 = Up(base_filters * 16 + base_filters * 8, base_filters * 8)
        
        # Output
        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder (first column)
        x00 = self.inc(x)      # X[0,0]
        x10 = self.down1(x00)  # X[1,0]
        x20 = self.down2(x10)  # X[2,0]
        x30 = self.down3(x20)  # X[3,0]
        x40 = self.bridge(x30) # X[4,0]
        
        # Decoder with dense connections
        # Second column
        x01 = self.up_01(x10, x00)  # X[0,1]
        x11 = self.up_12(x20, x10)  # X[1,1]
        x21 = self.up_23(x30, x20)  # X[2,1]
        x31 = self.up_34(x40, x30)  # X[3,1]
        
        # Third column
        x02 = self.up_02(x11, torch.cat([x00, x01], dim=1))  # X[0,2]
        x12 = self.up_13(x21, torch.cat([x10, x11], dim=1))  # X[1,2]
        x22 = self.up_24(x31, torch.cat([x20, x21], dim=1))  # X[2,2]
        
        # Fourth column
        x03 = self.up_03(x12, torch.cat([x00, x01, x02], dim=1))  # X[0,3]
        x13 = self.up_14(x22, torch.cat([x10, x11, x12], dim=1))  # X[1,3]
        
        # Fifth column (final output)
        x04 = self.up_04(x13, torch.cat([x00, x01, x02, x03], dim=1))  # X[0,4]
        
        return self.outc(x04)


# Visual representation of the UNet++ architecture:
"""
UNet++ Architecture (X[i,j] notation):

Encoder (Column 0):
X[0,0] → X[1,0] → X[2,0] → X[3,0] → X[4,0]

Dense Skip Connections:
X[0,0] → X[0,1] → X[0,2] → X[0,3] → X[0,4] (output)
      ↘       ↘       ↘       ↘
X[1,0] → X[1,1] → X[1,2] → X[1,3]
      ↘       ↘       ↘  
X[2,0] → X[2,1] → X[2,2]  
      ↘       ↘
X[3,0] → X[3,1]
      ↘
X[4,0]

Each node X[i,j] receives input from:
- The node below (X[i+1,j-1]) via upsampling
- All nodes to its left in the same row (X[i,0] to X[i,j-1]) via concatenation
"""


class SimpleUNetpp(nn.Module):
    """Simplified UNet++ with 3 encoder levels (easier to understand)"""
    def __init__(self, n_channels=1, n_classes=1, base_filters=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        
        # Bridge
        self.bridge = Down(base_filters * 4, base_filters * 8)
        
        # UNet++ decoder nodes
        # Column 1
        self.up_01 = Up(base_filters * 2 + base_filters, base_filters)
        self.up_12 = Up(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up_23 = Up(base_filters * 8 + base_filters * 4, base_filters * 4)
        
        # Column 2
        self.up_02 = Up(base_filters * 2 + base_filters * 2, base_filters)
        self.up_13 = Up(base_filters * 4 + base_filters * 2 * 2, base_filters * 2)
        
        # Column 3 (output)
        self.up_03 = Up(base_filters * 2 + base_filters * 3, base_filters)
        
        # Output
        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x00 = self.inc(x)      # X[0,0]
        x10 = self.down1(x00)  # X[1,0]
        x20 = self.down2(x10)  # X[2,0]
        x30 = self.bridge(x20) # X[3,0]
        
        # UNet++ dense decoder
        # Column 1
        x01 = self.up_01(x10, x00)  # X[0,1]
        x11 = self.up_12(x20, x10)  # X[1,1]
        x21 = self.up_23(x30, x20)  # X[2,1]
        
        # Column 2
        x02 = self.up_02(x11, torch.cat([x00, x01], dim=1))  # X[0,2]
        x12 = self.up_13(x21, torch.cat([x10, x11], dim=1))  # X[1,2]
        
        # Column 3 (output)
        x03 = self.up_03(x12, torch.cat([x00, x01, x02], dim=1))  # X[0,3]
        
        return self.outc(x03)