import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import data
import os
import torch.optim as optim
from torch.autograd import Variable
from model import RNASSP
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

data_set = data.RNADataset('data_seq/', 'data_csv/')

batch_size = 1
train_dataset = DataLoader(
    dataset=data_set, num_workers=1, batch_size=batch_size, shuffle=True)

# 固定随机种子
np.random.seed(0)
torch.manual_seed(0)  # cpu设置随机种子
torch.cuda.manual_seed_all(0)  # 为所有gpu设置随机种子
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seq_L_padding = 300
d = 8
epochs = 10
model = RNASSP(d).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion1 = nn.MSELoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)


def train(epoch):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    for (i, train_data) in enumerate(train_dataset):
        optimizer.zero_grad()
        ori_seq = train_data[0].to(device).float()  # batch_size*L*4
        true_notation = train_data[1].to(device).float()  # batch_size*L*3
        seq_L = ori_seq.shape[1]  # sequence length
        struc_len = true_notation.shape[1]
        if seq_L == struc_len:
            ori_seq = Variable(ori_seq)
            true_notation = Variable(true_notation)
            base_mat = base_matrix_concat(ori_seq)
            notation_produced = model(base_mat)
            loss = criterion1(notation_produced, true_notation)
            print('=> Epoch[{}]({}/{}): train_loss:{:.4f} train_acc:{:.4f}'.format(
                epoch,
                i,
                len()
            ))
            loss.backward()
            optimizer.step()
            # print(ori_seq.shape)
            # print(base_mat.shape)
            # print(notation_matrix.shape)
            # print(true_notation.shape)
        else:
            continue


def base_matrix_concat(x):
    '''
    concat base_embed from 1*L*4 to 1*L*L*8
    '''
    L = x.shape[1]
    x2 = x
    x = x.unsqueeze(1)
    x2 = x2.unsqueeze(1)
    x = x.repeat(1, L, 1, 1)
    x2 = x2.repeat(1, L, 1, 1)
    mat = torch.cat([x, x2], -1)
    return mat

# def notation_matrix_concat(x):
#     '''
#     concat notation_embed form 1*L*3 to 1*L*L*3
#     '''
#     L = x.shape[1]
#     x = x.unsqueeze(1)
#     x = x.repeat(1,L,1,1)
#     return x


if __name__ == '__main__':
    writer = SummaryWriter('./log')
    for epoch in range(epochs):
        train(epoch)
