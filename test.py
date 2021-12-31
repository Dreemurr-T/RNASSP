import torch
from model import RNASSP
import data
from utils import Nus_p
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNASSP(d=8).to(device)
model.load_state_dict(torch.load('checkpoint_tRNA/model_epoch_9.pth'))

model.eval()

seq_path = 'tRNA_seq/'
struc_path = 'tRNA_stru/'
data_set = data.RNADataset(seq_path, struc_path)

batch_size = 1
test_dataset = DataLoader(
    dataset=data_set, num_workers=0, batch_size=batch_size)
# print(len(test_dataset))

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
    x1 = x.unsqueeze(1)
    x1 = x1.repeat(1, L, 1, 1)
    return x1

def test():
    with torch.no_grad():
        for(i,test_data) in enumerate(test_dataset):
            ori_seq = test_data[0].to(device).float()  # batch_size*L*4
            true_notation = test_data[1].to(device).float()  # batch_size*L*3
            # print(ori_seq.shape)
            # print(true_notation.shape)
            base_mat = base_matrix_concat(ori_seq)
            notation_mat = notation_matrix_concat(true_notation)
            notation_mat = notation_mat.permute(0,3,1,2)
           
            notation_mat_p,notation_p = model(base_mat)
            true_notation = true_notation.squeeze()
            ori_seq = ori_seq.squeeze()
            notation_p = F.softmax(notation_p, dim=1)
            # print(notation_p)
            print_seq(ori_seq, true_notation,notation_p)

def print_seq(ori_seq, true_notation,notation_p):
    seq = ''
    not_1 = ''
    not_2 = ''
    for i in torch.max(ori_seq,1)[1]:
        if i == 0:
            seq += 'C'
        elif i == 1:
            seq += 'G'
        elif i == 2:
            seq += 'U'
        elif i == 3:
            seq += 'A'

    for i in torch.max(true_notation,1)[1]:
        if i == 0:
            not_1 += ')'
        elif i == 1:
            not_1 += '('
        elif i == 2:
            not_1 += '.'
    not_2 = Nus_p(notation_p,seq)
    print(seq)
    print(not_1)
    print(not_2)


if __name__ =='__main__':
    test()
            