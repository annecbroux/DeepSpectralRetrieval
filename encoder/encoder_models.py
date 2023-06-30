from torch import nn
from torch.nn import functional as F
import torch

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
            nn.Conv2d(3,nb_channels,kernel_size=(3,6),padding=(1,3),padding_mode='replicate'), # 256
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2), padding = (0,0)), # 128
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.ReplicationPad2d((2,2,0,0)), # 132
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # 66
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.ReplicationPad2d((1,1,0,0)), # 68
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # 34
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # 17
            nn.ReLU(),

            nn.Conv2d(nb_channels, 1, kernel_size=(3,1), padding = (1,0), padding_mode='replicate') # 17
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return x
      


class Encoder4_w_Resblock(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,1),padding_mode='replicate'), # (100,252)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,252)
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # (100,84)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,42)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,5), padding_size=(1,2)), # (100,42)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,21)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,19)
            nn.Conv2d(nb_channels, 1, kernel_size=(1,3)) # (100,17)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x



class Encoder4_w_Resblock(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,1),padding_mode='replicate'), # (100,252)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,252)
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # (100,84)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,42)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,42)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,21)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,21)

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,19)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 1, kernel_size=(1,3)) # (100,17)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x


class Encoder4_w_Resblock_1(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,1),padding_mode='replicate'), # (100,252)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,252)
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # (100,84)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,42)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,42)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,21)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,21)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,19)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 1, kernel_size=(1,3)) # (100,17)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x


class Encoder5_w_Resblock(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,1),padding_mode='replicate'), # (100,252)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,252)
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # (100,84)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,42)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,42)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,21)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,21)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,3), padding_size=(1,1)), # (100,21)
            nn.ReLU(),
            

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,19)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 1, kernel_size=(1,3)) # (100,17)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x




class Encoder6_w_Resblock(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,1),padding_mode='replicate'), # (100,252)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,252)
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # (100,84)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,42)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,42)
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # (100,21)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(1,5), padding_size=(0,2)), # (100,21)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(1,3), padding_size=(0,1)), # (100,21)
            nn.ReLU(),
            

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,19)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 1, kernel_size=(1,3)) # (100,17)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    



class Encoder7_w_Resblock(nn.Module): # uses strided convolutions instead of pooling
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,1),padding_mode='replicate'), # (100,252)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,252)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,84)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,84)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,42)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,42)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,21)
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(1,5), padding_size=(0,2)), # (100,21)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(1,3), padding_size=(0,1)), # (100,21)
            nn.ReLU(),
            

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,19)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 1, kernel_size=(1,3)) # (100,17)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    


class Encoder8_w_Resblock(nn.Module): # latent vars go into channel dimension
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,6),padding=(1,2),padding_mode='replicate'), # (100,255)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,255)
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,85)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,85)
            nn.ReLU(),
            nn.ReplicationPad2d((1,1,0,0)), # (100,87)
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,87)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,29)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,27)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,5), padding_size=(1,2)), # (100,27)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,9)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,3), padding_size=(1,1)), # (100,9)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,3)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 17, kernel_size=(1,3)) # (100,1)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x





class Encoder8_w_Resblock_1(nn.Module): # latent vars go into channel dimension
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,6),padding=(1,2),padding_mode='replicate'), # (100,255)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,255)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,85)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,85)
            nn.ReLU(),
            nn.ReplicationPad2d((1,1,0,0)), # (100,87)
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,87)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,29)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,27)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,5), padding_size=(1,2)), # (100,27)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,9)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,3), padding_size=(1,1)), # (100,9)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,3)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 17, kernel_size=(1,3)) # (100,1)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x




class Encoder9_w_Resblock(nn.Module): # latent vars go into channel dimension
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,3),padding_mode='replicate'), # (100,256)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,256)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,128)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,128)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,4), stride = (1,4)), # (100,32)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,32)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,16)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,5), padding_size=(1,2)), # (100,16)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride=(1,2)), # (100,8)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,4)
            nn.ReLU(),
            
            nn.Conv2d(nb_channels, 17, kernel_size=(1,4)) # (100,1)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    



class Encoder10_w_Resblock(nn.Module): # latent vars go into channel dimension
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,7),padding=(1,3),padding_mode='replicate'), # (100,256)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,256)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,128)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,128)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,4), stride = (1,4)), # (100,32)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,32)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,16)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(1,5), padding_size=(0,2)), # (100,16)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride=(1,2)), # (100,8)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,2), stride = (1,2)), # (100,4)
            nn.ReLU(),
            
            nn.Conv2d(nb_channels, 17, kernel_size=(1,4)) # (100,1)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    



class Encoder11_w_Resblock(nn.Module): # latent vars go into channel dimension
        
    def __init__(self,nb_channels):
        super().__init__() #initial size (100,256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,6),padding=(1,2),padding_mode='replicate'), # (100,255)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,255)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,85)
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,85)
            nn.ReLU(),
            nn.ReplicationPad2d((1,1,0,0)), # (100,87)
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)), # (100,87)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,29)
            nn.ReLU(),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3)), # (100,27)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(1,5), padding_size=(0,2)), # (100,27)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,9)
            nn.ReLU(),
            ResBlock(nb_channels,kernel_size=(1,3), padding_size=(0,1)), # (100,9)
            nn.ReLU(),

            nn.Conv2d(nb_channels, nb_channels, kernel_size=(1,3), stride = (1,3)), # (100,3)
            nn.ReLU(),
            nn.Conv2d(nb_channels, 17, kernel_size=(1,3)) # (100,1)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x



class Encoder3_w_Resblock(nn.Module):
        
    def __init__(self,nb_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,6),padding=(1,2),padding_mode='replicate'), # 255
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # 85
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.ReplicatePad2d((0,0,2,2)), # 91
            nn.Conv2d(nb_channels, nb_channels, kernel_size=(3,2), padding = (1,0),padding_mode='replicate'), # 90
            nn.ReLU(),
            
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # 30
            nn.ReLU(),
            nn.ReplicatePad2d((0,0,1,1)), # 32
            nn.ReLU(),

            ResBlock(nb_channels,kernel_size=(1,7), padding_size=(0,3)),
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # 16
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(1,5), padding_size=(0,2)), # 16
            nn.ReLU(),
            
            nn.Conv2d(nb_channels, 1, kernel_size=(1,1)) # 16

            # nn.Conv2d(nb_channels, 1, kernel_size=(1,3), padding = (0,0), padding_mode='replicate') # 10
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return x


class Encoder3_w_Resblock_Separate(nn.Module): # this was not tested yet
        
    def __init__(self,nb_channels):
        super().__init__()
        self.first_layerX = nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1),padding_mode='replicate')
        self.first_layerKa = nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1),padding_mode='replicate')
        self.first_layerW = nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1),padding_mode='replicate')        

        self.encoder = nn.Sequential(
            nn.Conv2d(3,nb_channels,kernel_size=(3,5),padding=(1,0),padding_mode='replicate'), # 252
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # 84
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(3,7), padding_size=(1,3)),
            nn.AvgPool2d(kernel_size=(1,3), stride = (1,3)), # 28
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(1,7), padding_size=(0,3)),
            nn.AvgPool2d(kernel_size=(1,2), stride = (1,2)), # 14
            nn.ReLU(),
            
            ResBlock(nb_channels,kernel_size=(1,7), padding_size=(0,3)),
            nn.ReLU(),
            
            nn.Conv2d(nb_channels, 1, kernel_size=(1,3), padding = (0,0), padding_mode='replicate')

            # nn.Conv2d(nb_channels, 1, kernel_size=(1,3), padding = (0,0), padding_mode='replicate')
        )
        
    def encode(self, x): # this needs to be fixed
        (xX, xKa, xW) = torch.split(x,1,dim=1)
        xX = self.first_layerX(xX)
        xKa = self.first_layerKa(xKa)
        xW = self.first_layerW(xW)
        x = torch.cat((xX,xKa,xW),dim=1)
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