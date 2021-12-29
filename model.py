import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear


class ResNetblock(nn.Module):
    def __init__(self, d):
        super(ResNetblock, self).__init__()
        self.d = d
        self.myresblock = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=2*d,
                      kernel_size=3, padding=2, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(2*d),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=2*d, out_channels=4*d,
                      kernel_size=5, padding=4, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(4*d),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=4*d, out_channels=8*d,
                      kernel_size=5, padding=4, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(8*d),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=8*d, out_channels=4*d,
                      kernel_size=5, padding=4, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(4*d),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=4*d, out_channels=2*d,
                      kernel_size=5, padding=4, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(2*d),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=2*d, out_channels=d,
                      kernel_size=3, padding=2, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(d),
        )

    def forward(self, x):
        residual = x
        out = self.myresblock(x)
        out += residual
        return out


class RNASSP(nn.Module):
    def __init__(self, d):
        super(RNASSP, self).__init__()
        self.d = d
        self.conv1 = nn.Conv2d(
            in_channels=d, out_channels=4*d, kernel_size=3, padding=2, dilation=2)
        self.act1 = nn.ELU()
        self.norm1 = nn.BatchNorm2d(4*d)
        self.resblock1 = ResNetblock(4*d)
        # self.resblock2 = ResNetblock(4*d)
        # self.resblock3 = ResNetblock(4*d)
        self.biLSTM = nn.LSTM(32,64,3,bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.act2 = nn.ELU()
        self.norm2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128,3,1,1)
        # self.output = nn.Softmax(dim=2)

    def forward(self, seq):
        '''
        input: L*L*8
        output: L*3
        '''
        seq = seq.permute(0, 3, 1, 2)  # N*d*L*L
        seq = self.conv1(seq)  # N*4d*L*L
        seq = self.act1(seq)
        seq = self.norm1(seq)
        seq = self.resblock1(seq) # N*4d*L*L
        # seq = self.resblock2(seq)
        # seq = self.resblock3(seq)  
        seq = seq.permute(0,2,3,1)
        seq = seq.squeeze()
        seq,(h0,c0) = self.biLSTM(seq) #L*L*6
        seq = self.drop(seq)
        seq = seq.permute(2,0,1)
        seq = seq.unsqueeze(dim=0)
        seq = self.conv2(seq)  # N*3*L*L
        seq_mean = seq.squeeze()
        seq_mean = seq_mean.permute(1,2,0)
        seq_mean = torch.mean(seq_mean, dim=0)
        # seq_mean = self.output(seq_mean)
        # seq = torch.mean(seq,dim=0)
        # seq = torch.mean(seq,dim=1)
        return seq,seq_mean
