from genericpath import exists
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import data
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import RNASSP
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

seq_path = '5s_seq/'
struc_path = '5s_stru/'
checkpoint_path = 'checkpoint/'

os.makedirs(checkpoint_path,exist_ok=True)

data_set = data.RNADataset(seq_path, struc_path)

batch_size = 1
train_dataset = DataLoader(
    dataset=data_set, num_workers=4, batch_size=batch_size, shuffle=True)

# 固定随机种子
np.random.seed(0)
torch.manual_seed(0)  # cpu设置随机种子
torch.cuda.manual_seed_all(0)  # 为所有gpu设置随机种子
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seq_L_padding = 300
d = 8
epochs = 50
model = RNASSP(d).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
criterion1 = nn.MSELoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)

def loss_function(loss1,loss2):
    loss = loss1 + 10*loss2
    return loss


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
            notation_mat = notation_matrix_concat(true_notation)
            notation_mat = notation_mat.permute(0,3,1,2)
            notation_mat_p,notation_p = model(base_mat)
            true_notation = true_notation.squeeze()
            loss1 = criterion1(notation_mat_p, notation_mat) #MSE between the two matrixes
            loss2 = criterion2(notation_p, true_notation)
            loss = loss_function(loss1,loss2)
            notation_p = F.softmax(notation_p, dim=1)
            acc = cal_acc(notation_p, true_notation)
            print('=> Epoch[{}]({}/{}): train_loss:{:.4f} train_acc:{:.4f}'.format(
                epoch,
                i,
                len(train_dataset),
                loss.item(),
                acc
            ))
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                writer.add_scalar('Train/Loss', loss, (epoch*len(train_dataset))+i)
                writer.add_scalar('Train/Acc', acc, (epoch*len(train_dataset))+i)
        else:
            continue
    scheduler.step()


def cal_acc(notation_produced, true_notation):
    acc = 0
    true_notation = true_notation
    prediction = notation_produced
    # prediction = torch.round(prediction)
    # print(prediction.shape)
    # print(true_notation.shape)
    max_index_true = torch.max(true_notation,1)[1]
    max_index_pre = torch.max(prediction,1)[1]
    for i in range(prediction.shape[0]):
        if max_index_pre[i] == max_index_true[i]:
            acc += 1
    acc /= prediction.shape[0]
    return acc


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


def notation_matrix_concat(x):
    '''
    concat notation_embed form 1*L*3 to 1*L*L*3
    '''
    L = x.shape[1]
    x = x.unsqueeze(1)
    x = x.repeat(1, L, 1, 1)
    return x


def save_checkpoint(epoch):
    if epoch % 10 == 0 or epoch == epochs-1:
        torch.save(model.state_dict(), checkpoint_path+'model_epoch_%s.pth' % epoch)


if __name__ == '__main__':
    writer = SummaryWriter('./log')
    for epoch in range(epochs):
        train(epoch)
        save_checkpoint(epoch)
