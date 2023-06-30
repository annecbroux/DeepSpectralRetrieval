from torch import nn
import torch
from torch.nn import functional as F

    
class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, padding_size, batch_normalization=True):
        super().__init__()
        self.batch_normalization = batch_normalization
        self.conv1 = nn.Conv1d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = padding_size, padding_mode='replicate')
        
        self.bn1 = nn.BatchNorm1d(nb_channels)

        self.conv2 = nn.Conv1d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = padding_size, padding_mode='replicate')

        self.bn2 = nn.BatchNorm1d(nb_channels)

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
    

    
    
class Decoder_256_wResblock_seq3_1(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=5,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=9)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
class Decoder_256_wResblock_seq4(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=3)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
class Decoder_256_wResblock_seq4_1(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=3)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
class Decoder_256_wResblock_seq5(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=5,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,nb_channels, kernel_size=5),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=6,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
class Decoder_256_wResblock_seq5_1(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1), #61
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=5,stride=1), #65
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True), #130
            nn.Conv1d(nb_channels,nb_channels, kernel_size=5), #126
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=6,stride=1), #131
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
   
class Decoder_256_wResblock_seq5_2(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=5,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,nb_channels, kernel_size=5),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=6,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x


class Decoder_256_wResblock_seq6(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )

        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=2),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=2,stride=2),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Conv1d(nb_channels,nb_channels, kernel_size=5)
        )
    
    
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
class Decoder_256_wResblock_seq6_1(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )

        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=2),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv1d(nb_channels,nb_channels, kernel_size=5)
        )
    
    
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x

    
    
    
class Decoder_256_wResblock_seq7_1(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=5,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=9)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x


    
    
class Decoder_256_wResblock_seq8_1(nn.Module): # works with n_hidden_units = 60
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1), #61
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=5,stride=1), #65
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True), #130
            nn.Conv1d(nb_channels,nb_channels, kernel_size=5), #126
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1), #129
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True), #258
            nn.Conv1d(nb_channels,1, kernel_size=3) #256
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    


class Decoder_256_wResblock_seq9_1(nn.Module): # works with n_hidden_units = 75
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1), #76
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1), #79
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1), #82
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=1.5, mode='linear',align_corners=True), #123
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1), #125
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True), #250
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1), #252
            nn.ReLU(),
            nn.ReplicationPad1d(2), #256
            nn.Conv1d(nb_channels,1, kernel_size=1) #256
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    


class Decoder_256_wResblock_seq10_1(nn.Module): # works with n_hidden_units = 75
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=1,stride=1), #75
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            # ResBlock(nb_channels, kernel_size=3, padding_size=1),
            # nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1), #77
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1), #79
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            # ResBlock(nb_channels, kernel_size=3, padding_size=1),
            # nn.ReLU(),
            nn.Upsample(scale_factor=1.5, mode='linear',align_corners=True), #119
            #nn.Conv1d(nb_channels,nb_channels, kernel_size=3), #115
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1), #121
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True), #242
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=3,stride=1), #244
            nn.ReLU(),
            nn.ReplicationPad1d(7), #256
            nn.Conv1d(nb_channels,1, kernel_size=1) #256
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
