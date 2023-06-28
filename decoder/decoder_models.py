from torch import nn
import torch
from torch.nn import functional as F

class Decoder_512_w_avgpool(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.fc1 = nn.Linear(n_input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_hidden_units)

        self.convt4 = nn.ConvTranspose1d(1, nb_channels, kernel_size=9,stride=2,padding=5)
        self.convt3 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=10,stride=2,padding=5)
        self.convt2 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=15,stride=2,padding=5)
        self.convt1 = nn.ConvTranspose1d(nb_channels, 1, kernel_size=8,stride=1,padding=1)
        self.avpool = nn.AvgPool1d(2)
        
    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(1)
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt2(x))
        x = self.convt1(x)
        x = self.avpool(x)
        return x


class Decoder_256(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.fc1 = nn.Linear(n_input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_hidden_units)

        self.convt4 = nn.ConvTranspose1d(1, nb_channels, kernel_size=7,stride=1,padding=2)
        self.convt3 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=5)
        self.convt2 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=6)
        self.convt1 = nn.ConvTranspose1d(nb_channels, 1, kernel_size=7,stride=1,padding=2)

    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(1)
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt2(x))
        x = self.convt1(x)
        return x    
    
    
class Decoder_256_conv(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.fc1 = nn.Linear(n_input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_hidden_units)

        self.convt4 = nn.ConvTranspose1d(1, nb_channels, kernel_size=7,stride=1,padding=2)
        self.convt3 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=5)
        self.convt2 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=6)
        self.convt1 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=9,stride=1,padding=2)
        self.conv = nn.Conv1d(nb_channels,1, kernel_size=3)
        
    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(1)
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt1(x))
        x = self.conv(x)
        return x
    
class Decoder_256_conv2(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.fc1 = nn.Linear(n_input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_hidden_units)

        self.convt4 = nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1,padding=1)
        self.convt3 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=2)
        self.convt2 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=2)
        self.convt1 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=1,padding=2)
        self.conv = nn.Conv1d(nb_channels,1, kernel_size=10)
        
    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(1)
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt1(x))
        x = self.conv(x)
        return x
    
class Decoder_256_wavpool(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.fc1 = nn.Linear(n_input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_hidden_units)

        self.convt4 = nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1,padding=1)
        self.convt3 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=3)
        self.convt2 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=3)
        self.convt1 = nn.ConvTranspose1d(nb_channels, 1, kernel_size=7,stride=1,padding=2)
        self.avpool = nn.AvgPool1d(3,stride=1)
        
    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(1)
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt1(x))
        
        x = self.avpool(x)
        return x
    
    
class Decoder_256_fc25(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.fc1 = nn.Linear(n_input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_hidden_units)

        self.convt5 = nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1,padding=1)
        self.convt4 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=1,padding=1)
        self.convt3 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=10,stride=2,padding=2)
        self.convt2 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=9,stride=2,padding=2)
        self.convt1 = nn.ConvTranspose1d(nb_channels, 1, kernel_size=8,stride=2,padding=2)

    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(1)
        x = F.relu(self.convt5(x))
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt2(x))
        x = self.convt1(x)
        return x
        
class Decoder_256_conv2_seq(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
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
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=2,padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=1,padding=2),
            nn.ReLU(),
            nn.Conv1d(nb_channels,1, kernel_size=10)
        )
        
    def forward(self, x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x) 
        return x
    
    
class Decoder_256_conv3(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.fc1 = nn.Linear(n_input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_hidden_units)

        self.convt4 = nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1)
        self.convt3 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear',align_corners=True)
        self.convt2 = nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear',align_corners=True)
        self.conv = nn.Conv1d(nb_channels,1, kernel_size=7)
        
    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(1)
        
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt3(x))
        
        x = self.upsample(x)
        x = F.relu(self.convt2(x))

        x = self.upsample(x)
        x = self.conv(x)
        return x
    
    
class Decoder_256_conv3_seq(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
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
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self, x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
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
    
class Decoder_256_wResblock_seq(nn.Module):
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
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
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    


class Decoder_256_wResblock_seq1(nn.Module):
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
class Decoder_256_wResblock_seq_dropout(nn.Module):
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Dropout(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
class Decoder_256_wResblock_seq2(nn.Module):
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
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
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
class Decoder_256_wResblock_seq3(nn.Module):
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
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1),
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
    
    
class Decoder_256_wResblock_seq3_1(nn.Module):
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
    
    
class Decoder_256_wResblock_seq4(nn.Module):
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
    
    
class Decoder_256_wResblock_seq4_1(nn.Module):
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
    
    
class Decoder_256_wResblock_seq5(nn.Module):
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
    
    
class Decoder_256_wResblock_seq5_1(nn.Module):
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
    
   
class Decoder_256_wResblock_seq5_2(nn.Module):
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


class Decoder_256_wResblock_seq6(nn.Module):
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
    
    
class Decoder_256_wResblock_seq6_1(nn.Module):
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

    
class Decoder_256_wResblock_160_seq(nn.Module):
    def __init__(self,nb_channels, n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels
        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.ConvTranspose1d(1, nb_channels, kernel_size=3,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=7,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=8,stride=1),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=7,stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            nn.Conv1d(nb_channels,1, kernel_size=3)
        )
        
    def decode(self,x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
class Decoder_256_w3Resblock_noConvT(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels

        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.Conv1d(1, nb_channels, kernel_size=3,stride=1,padding=1,padding_mode='replicate'),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            
            nn.Conv1d(nb_channels, nb_channels, kernel_size=5,stride=1,padding=2,padding_mode='replicate'),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            
            
            nn.Conv1d(nb_channels, nb_channels, kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            
            
            nn.Conv1d(nb_channels, nb_channels, kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self, x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x
    
    
    
    
class Decoder_256_wResblock_noConvT(nn.Module):
    def __init__(self,nb_channels,n_hidden_units, n_input_features):
        super().__init__()
        self.nb_channels = nb_channels

        self.linear_sequence = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        
        self.conv_sequence = nn.Sequential(
            nn.Conv1d(1, nb_channels, kernel_size=3,stride=1,padding=1,padding_mode='replicate'),
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            
            nn.Conv1d(nb_channels, nb_channels, kernel_size=5,stride=1,padding=2,padding_mode='replicate'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear',align_corners=True),
            
            
            nn.Conv1d(nb_channels, nb_channels, kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(nb_channels,1, kernel_size=7)
        )
        
    def decode(self, x):
        x = self.linear_sequence(x)
        x = x.unsqueeze(1)
        x = self.conv_sequence(x)
        return x    
    
    
    
class Decoder_256_wResblock_seq7_1(nn.Module):
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


    
    
class Decoder_256_wResblock_seq8_1(nn.Module):
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
            # ResBlock(nb_channels, kernel_size=3, padding_size=1),
            # nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=5,stride=1), #65
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            # ResBlock(nb_channels, kernel_size=3, padding_size=1),
            # nn.ReLU(),
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
    


class Decoder_256_wResblock_seq9_1(nn.Module):
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
            # ResBlock(nb_channels, kernel_size=3, padding_size=1),
            # nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1), #79
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(nb_channels, nb_channels, kernel_size=4,stride=1), #82
            nn.ReLU(),
            ResBlock(nb_channels, kernel_size=3, padding_size=1),
            nn.ReLU(),
            # ResBlock(nb_channels, kernel_size=3, padding_size=1),
            # nn.ReLU(),
            nn.Upsample(scale_factor=1.5, mode='linear',align_corners=True), #123
            #nn.Conv1d(nb_channels,nb_channels, kernel_size=3), #115
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
    


class Decoder_256_wResblock_seq10_1(nn.Module):
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
    