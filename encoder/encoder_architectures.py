"""
Contains the architecture of the encoder models used (most recent version)
"""

from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, padding_size, batch_normalization=True):
        super().__init__()
        self.batch_normalization = batch_normalization
        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = padding_size, padding_mode='replicate')
        
        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = padding_size, padding_mode='replicate')

        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization:
            y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization:
            y = self.bn2(y)
        y = y + x

        return y    
    
    
    
class Encoder1_w_Resblock(nn.Module):
    def __init__(self,nb_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2,nb_channels,kernel_size=(3,7),padding=(1,0),padding_mode='replicate'),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)),
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.ReLU(),
            
            nn.Conv2d(nb_channels, 1, kernel_size=(3,3), padding = (1,1), padding_mode='replicate')
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return x
        
        
def constrain_custom_latent_variables(y, norm_min, norm_max):
    y2 = y-norm_min
    y2[y2<0]=0
    y3 = y2+norm_min
    y4 = y3-norm_max
    y4[y4>0]=0
    y5 = y4+norm_max
    return y5.float()
