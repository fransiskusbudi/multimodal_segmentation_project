# models/unet.py

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """[Conv3d => BatchNorm => ReLU] x 2"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale MRI/CT).
        out_channels (int): Number of output channels (1 for binary, >1 for multi-class).
        features (list): List of feature sizes for each encoder/decoder block.
        output_activation (nn.Module or None): Optional activation (e.g., nn.Sigmoid, nn.Softmax(dim=1)).
        dropout_rate (float): Dropout rate for regularization (default: 0.1).
    """
    def __init__(self, in_channels=1, out_channels=1, 
                features=[16, 32, 64, 128],  
                #features=[32, 64, 128, 256], 
                 output_activation=None, dropout_rate=0.1):
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        
        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate)
        
        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature*2, feature, dropout_rate))
        
        # Final output layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                # Fix size mismatch due to rounding
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)
        
        x = self.final_conv(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
