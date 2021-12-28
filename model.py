import torch
import torch.nn as nn

class ResNetblock(nn.Module):
    def __init__(self,d):
        super(ResNetblock, self).__init__()
        self.d = d
        self.myresblock = nn.Sequential(
            nn.ELU(),
            nn.BatchNorm2d(d),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, padding=2, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(d),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=5, padding=4, dilation=2),
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
        self.conv1 = nn.Conv2d(in_channels=d, out_channels=4*d, kernel_size=3, padding=2, dilation=2)
        self.resblock = ResNetblock(4*d)
        self.act1 = nn.ELU()
        self.norm1 = nn.BatchNorm2d(4*d)
        self.biLSTM = nn.LSTM(32,3,2,bidirectional=True)
        # self.act2 = nn.ELU()
        # self.norm2 = nn.BatchNorm2d(6)
        self.fc = nn.Linear(6,3)
        self.output = nn.Softmax(dim=2)
        
    def forward(self, seq):
        '''
        input: L*L*8
        output: L*3
        '''
        seq = seq.permute(0, 3, 1, 2)  # N*d*L*L
        seq = self.conv1(seq) # N*4d*L*L
        for i in range(16):
            seq = self.resblock(seq)
            seq = self.act1(seq)
            seq = self.norm1(seq)
        seq_mean = torch.mean(seq,dim=3) # N*4d*L
        seq_mean = seq_mean.permute(2,0,1) # L*N*32
        seq_mean,(h0,c0) = self.biLSTM(seq_mean) #L*N*6
        # seq = torch.tensor(seq)
        # print(seq.shape)
        # seq = seq.permute(2,0,1) #6*L*L
        # seq = seq.unsqueeze(0)
        # seq = self.act2(seq)
        # seq = self.norm2(seq)
        # seq = seq.squeeze(0)
        # seq = seq.permute(1,2,0)
        seq_mean = self.fc(seq_mean) #L*N*3
        seq_mean = seq_mean.permute(1,0,2)
        seq = self.output(seq)
        # seq = torch.mean(seq,dim=0)
        # seq = torch.mean(seq,dim=1)
        return seq
